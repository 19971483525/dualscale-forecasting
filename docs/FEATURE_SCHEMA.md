# Prepared data schema

One row is one observed calendar day, sorted and unique by `date` (ISO date).
The audited local observations span 2024-01-01 through 2025-03-31, 456 days.

| Fields | Meaning | Prepared internal space | Physical reporting |
|---|---|---|---|
| `q_00` ... `q_95` | Quarter-hourly grid-side power, in chronological order | original 10^4 kW /1000 | multiply by10 to GW |
| `daily_load` | Grid-side daily total energy; main forecasting target | original 10^4 kWh /1000 | multiply by10 to GWh |
| `load_peak`, `load_valley`, `load_mean_96`, `load_std_96` | Daily statistics of the 96 power samples; SD uses ddof1 | internal power | multiply by10 to GW |
| `sales_total`, `all_industry_total`, sector fields below | Sales-side daily electricity, with overlapping hierarchical aggregates | original 10^4 kWh /1000 | multiply by10 to GWh |
| `sales_gap` | Grid energy minus sales energy | internal energy | multiply by10 to GWh |
| `sales_ratio` | Grid energy / sales energy | dimensionless | dimensionless |
| `temp_min`, `temp_avg`, `temp_max` | Wuhan daily temperature observations | degrees Celsius | degrees Celsius |
| `weekday_sin`, `weekday_cos`, `dayofyear_sin`, `dayofyear_cos`, `month_sin`, `month_cos`, `is_weekend` | Calendar covariates | dimensionless | dimensionless |

The grid-side energy/power relationship is
`daily_load = 0.25 * sum(q_00, ..., q_95)` in consistent internal units.
Sales totals and grid totals are complementary measurements with different
accounting definitions, not forced to be equal.

The 23 sales-side graph nodes in their audited order are:

```text
sales_total, all_industry_total, primary_industry, secondary_industry,
tertiary_industry, residential_total, urban_residential, rural_residential,
agriculture, industry, construction, transport_post, info_software,
wholesale_retail, hotel_catering, finance, real_estate, business_service,
public_service, high_energy, high_tech_equipment,
consumer_goods_manufacturing, other_manufacturing
```

These are not 23 disjoint sectors, and summing all nodes double-counts
aggregates. The 153-channel daily flat vector contains 96 profile values,
15 intraday-summary/calendar/weather values, 23 sales values, and 19 global
values. This deliberate overlap is retained from the original full model.

All input windows end on day t−1. Only the seven calendar fields of day t
are appended at the forecast origin; measured day-t weather, sales and load
are excluded. Normalizers are fitted on the training samples only. Main energy
targets use a normalized change from the previous day's energy, except the
explicit direct-target ablation. Auxiliary labels are used only in losses.

## Availability boundary

Every model input ends on the observation date t-1. Target-day calendar values
are deterministically known in advance. The weekend flag identifies Saturdays
and Sundays; scenario analysis separately handles official makeup workdays.
Actual source release and later-revision timestamps are not available. The
zero-lag retrospective protocol is an assumption about availability, and E09
tests simulated delays rather than a verified operational release schedule.
