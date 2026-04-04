# Workflow классификации жалоб (LangChain)

Проект сделан в формате "понятный notebook + переиспользуемый Python-модуль".
Все настройки, включая LLM, задаются в обычном JSON-конфиге.

## 1) Что такое LangChain в этом проекте

- `ChatPromptTemplate` — шаблон промпта с переменными (`{text}`, `{issues_table}`).
- `prompt | llm` — LCEL-цепочка: сначала подстановка переменных, потом запрос к модели.
- `with_structured_output(PydanticSchema)` — строгий JSON-выход по схеме.
- `chain.invoke(...)` — один запрос.
- `chain.batch([...])` — массовая обработка батчем.

Вся эта логика собрана в `complaints_workflow_pipeline.py` с комментариями.

## 2) Где задаются настройки

Для каждой пары продукт/категория создаётся свой JSON-конфиг в `configs/`.
Примеры:
- `configs/credit_card_retail.json` — Кредитная карта / Обслуживание по карте.
- `configs/debit_card_retail.json` — Дебетовая карта / Обслуживание по карте.

В конфиге находятся:
- `product_id` / `product_category` — значения для фильтрации входных данных (должны совпадать со значениями в таблице);
- `prompts_dir` — корневая папка промптов (общие для всех пар);
- пути к таксономиям;
- блок `llm` (ключ, модель, base_url, temperature, timeout);
- блок `runtime` (batch_size, max_concurrency, retries, backoff).

## 3) Структура папок

```
prompts/
  stage0_extract.txt              # общие промпты (одинаковые для всех пар)
  stage0_taxonomy.txt
  stage1_classification.txt
  judge_issues.txt
  judge_requests.txt
  judge_legacy.txt
  Кредитная карта/
    Обслуживание по карте/
      product_context.txt          # контекст продукта (уникален для каждой пары)
  Дебетовая карта/
    Обслуживание по карте/
      product_context.txt

taxonomies/
  Кредитная карта/
    Обслуживание по карте/
      issues.json                  # базовые справочники (заполняются после этапа 0b)
      requested_actions.json

results/
  Кредитная карта/
    Обслуживание по карте/
      stage0_extract/              # чанки parquet
      stage0_taxonomy_issues.json  # сгенерированные таксономии
      stage0_taxonomy_requested_actions.json
      stage1_classification/
      judge_issues/
      judge_requests/
```

Контекст продукта (`product_context.txt`) автоматически подставляется во все промпты через переменную `{product_context}`. Для новой пары достаточно создать только этот файл — промпты переиспользуются.

## 4) Схема workflow (этапы 0/1/2)

1. Вход: таблица с колонкой `text` (опционально `product_id`, `product_category`).
2. Этап 0a: извлечение `issues` и `requested_actions` в сыром виде.
3. Этап 0b: генерация двух отдельных таксономий:
   - `stage0_taxonomy_issues.json`
   - `stage0_taxonomy_requested_actions.json`
4. Ручная правка (если нужно): файлы редактируются прямо в `results/<pair>/`.
5. Этап 1: классификация по файлам таксономий из `results/<pair>/`.
6. Этап 2:
   - judge по `issues` (boolean для `category` и `sub_category`);
   - judge по `requested_actions` (отдельная цепочка);
   - опционально legacy judge (`decision`).
7. Отчеты: доли корректности глобально и по разрезам продукта/категории.

## 5) Структура notebook

`LangChain_workflow_классификация_жалоб.ipynb`:

1. Мини-справка по LangChain.
2. Загрузка конфига.
3. Создание LLM и чтение `product_context.txt`.
4. Сборка chain-объектов.
5. Загрузка и подготовка данных (в LLM передается только `text`).
6. Запуск этапа 0a.
7. Запуск этапа 0b (генерация issues/requested_actions таксономий отдельно).
8. Ручная правка файлов в `results/<pair>/` (опционально) и перечитывание таксономий в цепочки.
9. Запуск этапа 1.
10. Judge issues + judge requested_actions.
11. Метрики качества.

## 6) Зависимости

Смотри `requirements.txt`:
- `langchain-core`
- `langchain-openai`
- `pydantic`
- `pandas`
- `pyarrow`
- `jupyter`

## 7) Быстрый запуск

1. Установи зависимости:
   - `pip install -r requirements.txt`
2. Создай или выбери JSON-конфиг в `configs/` (например `credit_card_retail.json`).
3. Укажи в конфиге `llm.api_key` и `input_path`.
4. Создай файл `prompts/<product_id>/<product_category>/product_context.txt` с описанием продукта.
5. В notebook укажи путь к конфигу в переменной `CONFIG_PATH`.
6. Запусти `LangChain_workflow_классификация_жалоб.ipynb` сверху вниз.
7. После этапа 0b при необходимости поправь файлы таксономий в `results/<pair>/`, затем запусти следующую ячейку.

## 8) Чеклист валидации

- [ ] В prompt передается только `text`.
- [ ] Этап 0a отдает валидный JSON с `issues/requested_actions`.
- [ ] Этап 0b создает раздельные таксономии для issues и requested_actions.
- [ ] Перед этапом 1 (при необходимости) вручную отредактированы файлы в `results/<pair>/`.
- [ ] Этап 1 использует точные строки `category/sub_category` из итоговых файлов таксономии.
- [ ] Judge по `issues` и `requested_actions` запускается независимо.
- [ ] Длинные прогоны идут батчами, чанки сохраняются в parquet.
- [ ] Ретраи включены, HTTP timeout не выше 60 секунд.
