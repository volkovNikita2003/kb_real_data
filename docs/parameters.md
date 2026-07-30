# Входные и фактически использованные параметры

## Общие правила

Все входные параметры хранятся в YAML. Поддерживается `schema_version: 1`.
Версия относится к формату файла, а не к версии программы или эксперимента.

Правила чтения:

- неизвестное поле считается ошибкой;
- отсутствие обязательного поля считается ошибкой;
- строки, числа и логические значения не преобразуются друг в друга;
- целое число допустимо там, где ожидается вещественное;
- логическое значение не считается числом;
- `NaN`, положительная и отрицательная бесконечность запрещены;
- повторяющиеся ключи YAML считаются ошибкой;
- числовые выражения (`2 * 1000`, `1e3 / 2`) не вычисляются;
- все физические размеры должны быть положительными, если явно не указано
  иное;
- секция детектора означает, что детектор используется; отсутствие секции —
  что детектора нет.

Готовые примеры находятся в `examples/camera_only/` и
`examples/camera_line/`.

## `general.yaml`

Обязательны `schema_version`, непустая секция `detectors` и
`instrument.detector_position`.

```yaml
schema_version: 1
detectors:
  camera:
    width_px: 2592
    height_px: 1944
    pixel_width_m: 0.0000018965517241379312
    pixel_height_m: 0.0000018965517241379312
  line_sensor:
    pixel_count: 3643
    pixel_width_m: 0.000008
    pixel_height_m: 0.0002
instrument:
  detector_position: new
  wavelength_um: 0.633
  focal_length_um: 400000.0
```

`detector_position` принимает `old` или `new` и выбирает геометрию алгоритма
калибровки. Для экспериментов до изменения положения используется `old`, для
положения от 17.07.26 — `new`. Длина волны и фокусное расстояние необязательны;
показанные значения являются значениями по умолчанию.
Все поля физических параметров детекторов необязательны; в примере показаны
их значения по умолчанию. Размеры задаются непосредственно в метрах. Это
исключает изменение двоичного значения числа из-за промежуточного пересчёта
из микрометров.

## `calibration.yaml`

В первой версии поддерживается только автоматическая калибровка:

```yaml
schema_version: 1
mode: automatic
camera:
  hdr:
    mode: l2h
    difference_mode: after_hdr
    background_level: 12.0
    low_threshold: 10.0
    top_threshold: 240.0
line_sensor:
  pinhole_position_m: 0.0002
  signal_position_m: 0.0203
```

Набор секций детекторов обязан совпадать с `general.yaml`. Для линейки поля
`pinhole_position_m` и `signal_position_m` обязательны. Остальные настройки
внутри секций имеют значения по умолчанию:

| Поле | Камера | Линейка |
|---|---:|---:|
| `pinhole_diameter_um` | 200.0 | 200.0 |
| `gaussian_sigma_px` | 20.0 | 3.0 |
| `correct_pixel_size` | `true` | — |
| `hdr.mode` | `l2h` | — |
| `hdr.difference_mode` | `after_hdr` | — |
| `hdr.background_level` | 12.0 | — |
| `hdr.low_threshold` | 10.0 | — |
| `hdr.top_threshold` | 240.0 | — |
| `time_offset_us` | — | 2.0 |

Режимы ручной калибровки и автоматической калибровки с поправками оставлены
для следующей версии схемы.

## `darl.yaml`

Обязательны:

- секции присутствующих детекторов;
- коэффициенты преломления и поглощения частиц и тип частиц;
- три коэффициента преломления среды.

Секция камеры не имеет собственных полей. Для линейки
`logarithmic_radius_percent` по умолчанию равен 7.5. Секции присутствующих
детекторов должны быть указаны, а их набор
должен совпадать с `general.yaml`.

Поддерживаемые типы частиц: `sphere`, `rectangle`. Для прямоугольной частицы
`rectangle_aspect_ratio` по умолчанию равен 1.5.

Необязательные секции и их значения по умолчанию:

```yaml
laser:
  angle_deg: 0.0
  stage: 1
  power_w: 30.0
  polarization: parallel
signal:
  one_particle: false
  value_type: signal
particle_classes:
  split_type: 1
  min_diameter_nm: 100.0
  max_diameter_nm: 3500000.0
  mie_fraunhofer_boundary_nm: 10000.0
  mie_class_size_nm: 300.0
  fraunhofer_log_type: 1
  fraunhofer_log_percent: 7.5
```

## Параметры ожидаемого распределения измерения

Необязательный файл
`input_parameters/measurements/<имя измерения>.yaml` задаёт гипотезу о
распределении частиц конкретного измерения:

```yaml
schema_version: 1
expected_distribution:
  type: gaussian
  mean_nm: 15000.0
  sigma_nm: 3000.0
  particle_count: 1000000.0
```

Имя файла должно совпадать с именем существующего измерения. Сейчас
поддерживается распределение `gaussian`; все три числовых параметра обязательны
и должны быть положительными.

Отсутствие файла допустимо: аппаратная матрица и восстановление реального
сигнала от него не зависят. Файл нужен только для расчёта ожидаемого сигнала и
его сравнения с реальным.

## Параметры численного контроля

Настройки численного контроля качества не задаются во входном файле. Они
добавляются в `output/darl/quality_control/used-parameters.yaml`:

```yaml
quality_control:
  restoration_type: 1
  small_particle_boundary_nm: 300.0
  evaluation_type: 1
  class_frequency: 10
```

Этот контроль создаёт внутренние распределения по классам, моделирует их
сигналы и выполняет восстановление. Он не использует ожидаемые распределения
измерений и не является восстановлением реального сигнала.

## Профиль восстановления

Каждый файл `restore_profiles/<имя>.yaml` содержит независимый профиль.
Обязательна непустая секция `detectors`; профиль может использовать подмножество
детекторов эксперимента.

```yaml
schema_version: 1
detectors:
  camera:
    use_background: true
    hdr:
      mode: l2h
      difference_mode: after_hdr
      background_level: 12
      low_threshold: 10
      top_threshold: 240
      filtered: false
      gaussian_sigma: 5
  line_sensor:
    use_background: true
    signal_mode: 2
    time_offset_us: 2
solver:
  type: tikhonov_nnls
  regularization_order: 1
  regularization_alpha: best
  use_w_critical: false
  use_chahine: true
  use_concentration_correction: true
class_slice:
  drop_first: 10
  drop_last: null
```

`regularization_alpha` принимает `best` или положительное число.
`use_w_critical` включает отсечение малых весовых долей по legacy-порогу и по
умолчанию выключен для совпадения с эталонными восстановлениями.

Секции `camera.hdr` и параметры линейки необязательны. Допустимые режимы HDR:
`l2h`/`l2h_longest` и `after_hdr`/`per_exposure`. Пороговые значения должны
находиться в диапазоне от 0 до 255, причём `low_threshold < top_threshold`.
`line_sensor.signal_mode` принимает 1 или 2, а `time_offset_us` должен быть
неотрицательным. Все отсутствующие значения раскрываются в
`used-parameters.yaml`.

## Фактически использованные параметры

`CalibrationStageParameters` загружает только `general.yaml` и
`calibration.yaml`, поэтому калибровка не зависит от наличия и содержимого
`darl.yaml`. `ExperimentParameters` дополнительно загружает `darl.yaml` для
следующих этапов и проверяет согласованность детекторов. Методы
`effective_calibration()`, `effective_matrix()`,
`effective_quality_control()`, `effective_expected_signal()` и
`effective_restore()` создают полные словари соответствующих запусков. Они
включают значения по умолчанию и результаты предыдущих этапов.

CLI восстановления не загружает полный `ExperimentParameters`: его входами
служат только `general.yaml`, выбранный restore profile,
`calibration/result.yaml`, `darl/result.yaml` и данные выбранного измерения.
Для каждой пары он формирует собственный полный `used-parameters.yaml`, где
сохраняются нормализованный профиль, фактические экспозиции и пути сигналов,
оба входных результата предыдущих этапов и все раскрытые значения по
умолчанию. Поэтому изменение исходных `calibration.yaml` или `darl.yaml` после
публикации результатов не меняет вход восстановления.

Функция `write_used_parameters()` сохраняет такой словарь как безопасный YAML.
Существующий файл она не перезаписывает; управление `--force` и архивированием
реализовано в модуле `output.py`.

Техническая архитектура чтения, записи и проверки YAML описана в
[yaml-processing.md](yaml-processing.md). В частности, там объясняется
различие между проверкой одной YAML-схемы и валидацией структуры эксперимента.
