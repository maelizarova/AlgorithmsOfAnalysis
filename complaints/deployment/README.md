# Деплоймент классификации претензий

Папка `deployment` содержит Prefect flow для ежедневного скоринга новых претензий LLM-классификаторами.

## Что переносить

Для деплоймента нужна именно папка `deployment` целиком. Внутри уже лежат:

- `complaints_scoring_flow.py` — основной Prefect flow;
- `settings.json` — общие настройки Oracle, LLM и runtime;
- `configs/` — JSON-конфиги классификаторов и пример таксономии;
- `prompts/` — промпты для классификации и judge;
- `queries/get_data.sql` — SQL для получения претензий;
- `tasks/`, `models/`, `utils/`, `llm_pipeline.py` — код выполнения.

Unit-тесты не нужны для боевого переноса и удалены из папки.

## Как работает flow

`complaints_scoring_flow.py` читает все JSON-файлы из `configs/*.json`.
Один JSON-конфиг соответствует одному классификатору.
Общие параметры подключения и батчинга берутся из `deployment/settings.json`, поэтому их не нужно дублировать в каждом классификаторе.

Текст претензии читается из `cea.description_claim` в `queries/get_data.sql`.
Поле извлекается через `dbms_lob.substr(cea.description_claim, 2000, 1)`, чтобы не тянуть CLOB целиком через DB link `@siebel` и не ловить ORA-06502 на кириллице (лимит VARCHAR2 в SQL — 4000 байт).

По умолчанию:

- `report_date` — дата отчёта (по умолчанию вчера, `today - 1`);
- окно данных — `[report_date - lookback_days + 1, report_date + 1)`;
- `lookback_days = 1`, то есть скорятся претензии за `report_date` (вчера).

Если нужно перескорить несколько дней, передайте параметр `lookback_days`, например `3`.

## Результирующая таблица

Результат пишется в `ema_complaints_classification`.

Схема строк нормализованная:

- `claim_num` — номер претензии;
- `created` — дата создания претензии;
- `report_date` — дата отчёта (по умолчанию вчера);
- `product`, `theme`, `category` — продуктовые поля из источника;
- `classifier_name` — имя классификатора из конфига;
- `type` — `issue` или `req_action`;
- `class` — класс из таксономии;
- `sub_class` — подкласс из таксономии;
- `eval` — `true`, если judge подтвердил и класс, и подкласс.

Перед записью flow удаляет старые строки за тот же `report_date`, затем каждый классификатор делает append.

## Как добавить новый классификатор

1. Скопируйте `configs/auto_loan.json`.
2. Задайте уникальный `classifier_name`.
3. Заполните `product`, при необходимости `theme` и `category`.
4. Положите таксономии рядом с конфигом, например:
   - `configs/my_classifier/issues.json`;
   - `configs/my_classifier/requested_actions.json`.
5. В конфиге укажите пути:
   - `"taxonomy_issues_path": "my_classifier/issues.json"`;
   - `"taxonomy_requests_path": "my_classifier/requested_actions.json"`.
6. Если нужен отдельный контекст продукта, добавьте файл:
   - `deployment/prompts/<product>/product_context.txt`.

Oracle, LLM и `runtime` обычно менять в каждом классификаторе не нужно. Они лежат в `deployment/settings.json`.
Если для конкретного классификатора всё-таки нужен другой LLM или другой размер батча, можно добавить секцию `llm` или `runtime` прямо в его JSON — она переопределит общие настройки.

## Что такое `product`, `theme`, `category`

Эти поля приходят из SQL `deployment/queries/get_data.sql`:

```sql
pr.desc_text as product,
th.desc_text as theme,
cat.desc_text as category
```

- `product` — продукт претензии. Это обязательный фильтр классификатора.
- `theme` — тема претензии из Siebel-справочника `JET_SR_THEME`. Это опциональный фильтр.
- `category` — категория претензии из Siebel-справочника `JET_SR_CATEGORY`. Это опциональный фильтр.

Если `theme` или `category` в JSON-конфиге равны `null` или отсутствуют, фильтра по ним не будет.
Например, текущий `auto_loan.json` скорит все претензии с `product = "Автокредит"` независимо от темы и категории.

## Как подтягивается контекст продукта

Путь к `product_context.txt` в JSON-конфиге не задаётся отдельно.
Код ищет контекст автоматически внутри `deployment/prompts`.

Для конфига:

```json
{
  "product": "Автокредит",
  "theme": null,
  "category": null
}
```

будет использован файл:

```text
deployment/prompts/Автокредит/product_context.txt
```

Если в конфиге заполнены `theme` или `category`, код сначала пробует более детальные пути:

```text
deployment/prompts/<product>/<theme>/product_context.txt
deployment/prompts/<product>/<category>/product_context.txt
deployment/prompts/<product>/product_context.txt
```

Если файл контекста не найден, flow продолжит работу без контекста продукта.

## Секреты Prefect

В примере используются Secret blocks:

- `pass-space` — пароль Oracle для пользователя `analytics`;
- `llm-api-key` — ключ LLM API.

Если LLM-ключ не нужно хранить в Prefect Secret, можно указать его прямо в конфиге в поле `llm.api_key`.

## Ручной запуск

Перед полным запуском удобно открыть `check_flow_debug.ipynb`.
Он проходит тот же пайплайн по шагам и показывает промежуточные результаты:
SQL-выборку, фильтр, sample для LLM, результат классификации, judge и финальную таблицу.
По умолчанию notebook обрабатывает только несколько строк и не пишет результат в Oracle.

Из папки `deployment`:

```bash
python complaints_scoring_flow.py
```

Пример запуска с окном за 3 дня:

```python
from complaints_scoring_flow import run_all_classifiers

run_all_classifiers(report_date="2026-06-10", lookback_days=3)
```

## Зависимости

Список Python-зависимостей лежит в `deployment/requirements.txt`.
В окружении также должен быть доступен внутренний пакет `toolbox`, потому что он используется для создания Oracle engine так же, как в проекте мониторинга моделей.
