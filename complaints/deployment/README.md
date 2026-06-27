# Деплоймент классификации претензий

Папка `deployment` содержит Prefect flow для ежедневного скоринга новых претензий LLM-классификаторами.

## Что переносить

Для деплоймента нужна именно папка `deployment` целиком. Внутри уже лежат:

- `complaints_scoring_flow.py` — основной Prefect flow;
- `configs/` — JSON-конфиги классификаторов и пример таксономии;
- `prompts/` — промпты для классификации и judge;
- `queries/get_data.sql` — SQL для получения претензий;
- `tasks/`, `models/`, `utils/`, `llm_pipeline.py` — код выполнения.

Unit-тесты не нужны для боевого переноса и удалены из папки.

## Как работает flow

`complaints_scoring_flow.py` читает все JSON-файлы из `deployment/configs/*.json`.
Один JSON-конфиг соответствует одному классификатору.

По умолчанию:

- `score_date` — дата запуска flow;
- окно данных — `[score_date - lookback_days, score_date)`;
- `lookback_days = 1`, то есть скорятся претензии за вчера.

Если нужно перескорить несколько дней, передайте параметр `lookback_days`, например `3`.

## Результирующая таблица

Результат пишется в `ema_complaints_classification`.

Схема строк нормализованная:

- `claim_num` — номер претензии;
- `created` — дата создания претензии;
- `score_date` — дата скоринга;
- `product`, `theme`, `category` — продуктовые поля из источника;
- `classifier_name` — имя классификатора из конфига;
- `type` — `issue` или `req_action`;
- `class` — класс из таксономии;
- `sub_class` — подкласс из таксономии;
- `eval` — `true`, если judge подтвердил и класс, и подкласс.

Перед записью flow удаляет старые строки за тот же `score_date` и `classifier_name`, чтобы не плодить дубли при повторном запуске.

## Как добавить новый классификатор

1. Скопируйте `deployment/configs/auto_loan.json`.
2. Задайте уникальный `classifier_name`.
3. Заполните `product`, при необходимости `theme` и `category`.
4. Положите таксономии рядом с конфигом, например:
   - `deployment/configs/my_classifier/issues.json`;
   - `deployment/configs/my_classifier/requested_actions.json`.
5. В конфиге укажите пути:
   - `"taxonomy_issues_path": "my_classifier/issues.json"`;
   - `"taxonomy_requests_path": "my_classifier/requested_actions.json"`.
6. Если нужен отдельный контекст продукта, добавьте файл:
   - `deployment/prompts/<product>/product_context.txt`.

## Секреты Prefect

В примере используются Secret blocks:

- `pass-space` — пароль Oracle для пользователя `analytics`;
- `llm-api-key` — ключ LLM API.

Если LLM-ключ не нужно хранить в Prefect Secret, можно указать его прямо в конфиге в поле `llm.api_key`.

## Ручной запуск

Из корня проекта, где лежит папка `deployment`:

```bash
python -m deployment.complaints_scoring_flow
```

Пример запуска с окном за 3 дня:

```python
from deployment.complaints_scoring_flow import run_all_classifiers

run_all_classifiers(score_date="2026-06-10", lookback_days=3)
```

## Зависимости

Список Python-зависимостей лежит в `deployment/requirements.txt`.
В окружении также должен быть доступен внутренний пакет `toolbox`, потому что он используется для создания Oracle engine так же, как в проекте мониторинга моделей.
