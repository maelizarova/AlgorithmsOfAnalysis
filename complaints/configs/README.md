# Как заполнять JSON-конфиг

Для каждой пары продукт/категория создаётся отдельный JSON-файл.
Примеры:
- `credit_card_retail.json` — Кредитная карта / Обслуживание по карте.
- `debit_card_retail.json` — Дебетовая карта / Обслуживание по карте.

## Корневые поля

- `product_id` — название продукта, как в данных (например, `Кредитная карта`).
- `product_category` — категория продукта, как в данных (например, `Обслуживание по карте`).
- `input_path` — путь к входному файлу с обращениями (CSV или parquet).
- `prompts_dir` — корневая папка с общими prompt-файлами (обычно `"../prompts"`).
- `taxonomy_issues_path` — путь к справочнику проблем issues.
- `taxonomy_requests_path` — путь к справочнику запросов requested_actions.
- `text_column` — название колонки с текстом обращения во входных данных.
- `llm` — настройки LLM.
- `runtime` — настройки батчинга, ретраев и параллелизма.

Значения `product_id` и `product_category` используются для:
1. Фильтрации входных данных (если в таблице есть эти колонки).
2. Формирования `pair_key` (`product_id/product_category`) для папок в `results/`.
3. Поиска файла `product_context.txt` в `prompts_dir/product_id/product_category/`.

## Блок `llm`

- `api_key` — ключ API (если не требуется — поставить любую непустую строку, например `"not-needed"`).
- `base_url` — URL API-провайдера.
- `model` — имя модели.
- `temperature` — креативность генерации (обычно `0` для классификации).
- `timeout_seconds` — HTTP timeout (в коде ограничивается 60 сек).
- `model_kwargs` — дополнительные параметры модели (`top_p`, `top_k`, `max_tokens` и т.д.).

## Блок `runtime`

- `batch_size` — размер батча.
- `max_concurrency` — сколько запросов отправлять параллельно в `chain.batch`.
- `retries` — количество повторов при ошибке.
- `backoff_base_seconds` — базовая пауза для экспоненциального backoff.

## Контекст продукта

Для каждой пары в папке `prompts/<product_id>/<product_category>/` должен лежать файл `product_context.txt`.
Он содержит текстовое описание продукта и типичных тем жалоб.
Этот текст автоматически подставляется во все промпты через переменную `{product_context}`.
Если файла нет — промпты работают без контекста.

## Как добавить новую пару

1. Создай новый JSON в `configs/`.
2. Укажи `product_id` и `product_category` точно как в данных.
3. Создай файл `prompts/<product_id>/<product_category>/product_context.txt`.
4. Укажи путь к новому JSON в notebook в переменной `CONFIG_PATH`.
5. Запусти notebook — этап 0b сгенерирует таксономии автоматически.
