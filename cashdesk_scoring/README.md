# Ежедневный скоринг касс (Prefect)

Продакшен-пайплайн потребности касс: central SARIMA + иерархическая поправка ошибок, зануление финального прогноза только по расписанию/праздникам.

Старые research-ноутбуки не используются и не меняются.

## Файлы

| Файл | Назначение |
|---|---|
| `settings.json` | Параметры алгоритма и Oracle |
| `cashdesk_forecast.py` | Загрузка данных, SARIMA, adaptive q, schedule |
| `cashdesk_scoring_flow.py` | Prefect flow `ts_cashdesks_0826_v1` |
| `oracle_writer.py` | Delete по `score_date` + append |
| `schema_preds.sql` | DDL одной партиционированной таблицы |

## Параметры (зафиксированы)

- окно ошибок: **280** дней
- shrinkage: **10**
- `initial_quantile` / `target_breach_rate`: **0.13**
- lag факта: 2 дня, горизонт: 30 дней
- обучение: пропуски в ряде = `NaN`
- после прогноза: закрытые по графику/празднику → pred = 0

## Таблица `EMA_CASHDESK_PREDS`

Одна таблица, INTERVAL-партиции по `score_date`.

В каждой строке есть:
- финальный `atdtmco_ns_pred` / `atdtmco_saldo_turn_pred`
- `atdtmco_ns_pred_central` — для ошибок on-the-fly: `(fact - central) / scale`, где `scale` пересчитывается из фактов
- `quantile_used` — чтобы завтра взять вчерашний `q`
- `atdtmco_cashdesk_code`, `is_closed`

Отдельные таблицы ошибок и adaptive state не нужны.

## Запуск

1. Создать таблицу из `schema_preds.sql`.
2. (Рекомендуется) backfill истории ≥ 280 + lag дней.
3. Задеплоить/запустить flow:

```python
from cashdesk_scoring_flow import ts_cashdesks_0826_v1
ts_cashdesks_0826_v1(score_date="2026-08-12")
```

Или:

```bash
python cashdesk_scoring_flow.py
```


Повторный запуск за ту же дату идемпотентен: строки `score_date` удаляются и пишутся заново.

## One-time backfill

Перед первым prod-днём залить в `EMA_CASHDESK_PREDS` историю прошлых `score_date` с полями central/final/`quantile_used`/`is_closed`. Иначе пул ошибок и адаптивный квантиль стартуют «с нуля» (`q=0.13`, пустые residuals до накопления фактов).
