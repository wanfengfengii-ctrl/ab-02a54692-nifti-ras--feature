# nifti-sampler

神经影像质控平台的体数据抽查服务：按扫描仪 RAS 世界坐标对三维 NIfTI-1
体数据做三线性插值采样（`/api/nifti/sample`），并按 RAS 闭球统计标记点
周围的局部强度分布（`/api/nifti/sphere-stats`）。严格校验方向矩阵
（sform/qform）、字节序与强度缩放，避免核对到错误体素。纯 Python 标准库
实现，无第三方依赖。

## API

### `POST /api/nifti/sample`

`multipart/form-data`，包含：

| 部分 | 说明 |
| --- | --- |
| 文件部分（任意字段名，须带 `filename`） | 单个 NIfTI-1 `.nii` 文件，≤ 16 MiB |
| `points` 字段 | JSON 数组，1–256 个元素：`[{"id": 0, "point": [x, y, z]}, ...]` |

`id` 为 `[0, 2^31)` 内唯一整数；`point` 为三个有限数值（RAS 世界坐标，
单位与文件仿射一致，通常为 mm）。

```bash
curl -F "file=@vol.nii" \
     -F 'points=[{"id": 1, "point": [12.0, 26.0, 42.0]}]' \
     http://localhost:8000/api/nifti/sample
```

**成功响应（200）**，结果顺序与请求一致：

```json
{
  "transform": "sform",
  "results": [
    {"id": 1, "status": "ok", "voxel": [1.0, 2.0, 3.0],
     "intensity": 647.0, "transform": "sform"},
    {"id": 2, "status": "error",
     "error": {"code": "out_of_bounds", "message": "...",
               "voxel": [-5.0, -6.67, -7.5]}}
  ]
}
```

每个成功点给出：连续体素坐标（`voxel`）、插值强度（`intensity`，已应用
`scl_slope`/`scl_inter` 缩放）、所用变换来源（`transform`，
`sform` 或 `qform`）。越界点、非有限数据以**逐点错误**返回，不影响其余点。

### `POST /api/nifti/sphere-stats`

`multipart/form-data`，沿用与采样接口相同的文件限制（单个 NIfTI-1 `.nii`
≤ 16 MiB），另一部分为 `regions` JSON：

| 部分 | 说明 |
| --- | --- |
| 文件部分（任意字段名，须带 `filename`） | 单个 NIfTI-1 `.nii` 文件，≤ 16 MiB |
| `regions` 字段 | JSON 数组，1–32 个元素：`[{"id": 1, "center": [x, y, z], "radius": r}, ...]` |

`id` 为 `[0, 2^31)` 内唯一整数；`center` 为三个有限数值（RAS 世界球心），
`radius` 为正有限数值（世界空间半径，单位与仿射一致，通常 mm）。

裁决与统计规则：

- **世界空间闭球**：在所选 sform/qform 世界空间中，把每个整数体素中心经
  仿射正变换映射到 RAS 后裁决，欧氏距离 ≤ radius 即纳入（球面本体积入，
  含浮点容差）；候选范围由逆仿射行范数保守界定，再做精确二次距离判定。
- 球体可以越出影像边界，**只统计实际存在的体素中心**；一个中心都不覆盖时
  该区域报 `empty_region`。
- 单区纳入中心数上限 **65536**；第 65537 个候选中心将令该区域报
  `region_too_large`（另有 O(1) 内接盒下界快速失败，细网格上也不会全量扫描）。
- 强度继续沿用既有缩放规则：`scl_slope != 0` 时 `raw*slope+inter`，否则取
  原始 float 值；区域内出现非有限缩放强度（NaN/Inf）时该区域报
  `non_finite_data`，错误定位到具体体素。
- 结果保持请求顺序，逐区返回；空区域、超容量或坏体素**只使对应编号失败**，
  不遮蔽其余区域的统计。

```bash
curl -F "file=@vol.nii" \
     -F 'regions=[{"id": 1, "center": [12.0, 26.0, 42.0], "radius": 3.0}]' \
     http://localhost:8000/api/nifti/sphere-stats
```

**成功响应（200）**，结果顺序与请求一致：

```json
{
  "transform": "sform",
  "results": [
    {"id": 1, "status": "ok", "count": 123, "min": -10.0, "max": 2047.0,
     "mean": 314.2, "transform": "sform"},
    {"id": 2, "status": "error",
     "error": {"code": "empty_region", "message": "..."}},
    {"id": 3, "status": "error",
     "error": {"code": "non_finite_data", "message": "...",
               "voxel": [12, 7, 3]}}
  ]
}
```

逐区成功字段：`count`（纳入的实际体素中心数）、`min`/`max`/`mean`（缩放后
有限强度的最小、最大、算术平均）、`transform`（`sform` 或 `qform`）。

### 错误

文件/请求级错误返回 4xx，JSON 形如
`{"error": {"code", "message", "field"}}`，`field` 定位到头部字段或表单字段：

| code | status | field | 含义 |
| --- | --- | --- | --- |
| `invalid_multipart` | 400/415 | — | 表单结构非法 |
| `missing_file` / `multiple_files` | 400 | `file` | 文件部分缺失/多于一个 |
| `missing_points` / `invalid_points` | 400 | `points` | 坐标字段缺失、数量越界、id 重复/非法、坐标非有限（message 含点号） |
| `missing_regions` / `invalid_regions` | 400 | `regions` | regions 字段缺失、数量非 1–32、id 重复/非法、center 非有限、radius 非正有限（message 含编号） |
| `file_too_large` | 413 | `file` | 超过 16 MiB |
| `header_too_short` | 400 | `file` | 不足 348 字节头部 |
| `bad_sizeof_hdr` | 400 | `sizeof_hdr` | 两种字节序下都不是 348 |
| `unsupported_magic` | 400 | `magic` | 非 `n+1`（如 `.hdr/.img` 对的 `ni1`） |
| `invalid_dimensions` | 400 | `dim` | 非完整三维（`dim[0]!=3`、`dim[4..7]!=1`、维度非正） |
| `unsupported_datatype` | 400 | `datatype` | 仅接受 int16(4)/float32(16) |
| `bitpix_mismatch` | 400 | `bitpix` | 与 datatype 不一致 |
| `invalid_vox_offset` | 400 | `vox_offset` | 非有限整数或 < 352 |
| `payload_length_mismatch` | 400 | `file` | 载荷长度与头部不符（截断） |
| `trailing_bytes` | 400 | `file` | 文件含尾随字节 |
| `non_finite_scaling` | 400 | `scl_slope`/`scl_inter` | 缩放参数非有限 |
| `missing_affine` | 400 | `qform_code,sform_code` | 两种变换码均为 0 |
| `non_finite_affine` | 400 | `srow_*`/`quatern_*`/`qoffset_*` | 仿射参数非有限 |
| `invalid_pixdim` | 400 | `pixdim` | qform 所需体素尺寸非正/非有限 |
| `singular_affine` | 400 | `srow`/`qform` | 所选仿射不可逆 |

逐点错误（200 响应内）：`out_of_bounds`（逆变换后落在体素中心闭域
`[0, n-1]` 之外）、`non_finite_data`（插值邻域内缩放后数据非有限）。

逐区错误（200 响应内，不影响其余区域）：`empty_region`（闭球未覆盖任何
实际体素中心）、`region_too_large`（纳入中心超过 65536）、
`non_finite_data`（球内某体素缩放后强度非有限，错误附带该体素坐标）。

### `GET /healthz`

就绪探针：服务启动时对采样流水线做自检，通过后才返回
`200 {"status": "ok", "ready": true}`，否则 503。

## 接受的 NIfTI 子集与采样规则

- 单文件 `.nii`（magic `n+1`），完整三维：`dim[0]=3`、`dim[4..7]=1`、维度为正。
- 数据类型 int16 或 float32，bitpix 一致；大端/小端均可（按 `sizeof_hdr` 判定）。
- `vox_offset` 为 ≥352 的有限整数；载荷长度须与维度精确一致，不得有尾随字节。
- `scl_slope`/`scl_inter` 必须有限；`scl_slope == 0` 时按 NIfTI 约定不缩放。
- 仿射选择：**有效 sform（`sform_code>0`）优先于 qform**；两者皆缺、或所选
  仿射不可逆（行列式为 0/非有限）则拒绝。qform 按 nifti1_io 规则由四元数、
  `pixdim[1..3]` 与 `qfac = sign(pixdim[0])` 重建。
- 世界坐标经所选仿射的逆变换映射到连续体素坐标，须落在体素中心闭域
  `[0, n-1]`（各轴，含边界；另有 1e-6 体素的浮点容差）。对缩放后的 8 个
  邻近体素做三线性插值；边界轴固定到唯一端点（权重 1），零权重邻居不参与。

### 球体统计规则（sphere-stats）

- 球心与半径均在所选 sform/qform 的世界空间中给出；体素中心先经仿射
  **正变换**到 RAS，再以欧氏距离闭球裁决（表面计入，带与世界坐标量级成
  比例的浮点容差，覆盖 float32 头部 + 大平移下的抵消误差）。
- 候选体素盒由球心经逆仿射映射、半径乘逆仿射行范数给出（保守外盒，另加
  1e-6 体素），逐中心做精确世界空间二次距离判定；球可越界，只统计影像内
  实际存在的中心，故无需体素插值——读取的就是真实体素中心。
- 容量按几何成员数计：`count ≤ 65536`；恰为 65536 合法，超出则仅该区失败。
  内接盒（基于仿射 Frobenius 范数对最大奇异值的保守上界）成员数超过上限时
  O(1) 直接失败。
- 统计值均在缩放后的强度上计算；任一成员为 NaN/±Inf 时该区失败并定位到
  首个非有限体素，其余区域照常统计。

## 运行

本地（需 Python ≥ 3.11）：

```bash
PORT=8000 python -m app.server
```

Docker（宿主机端口用 `NIFTI_HOST_PORT` 配置，默认 8000）：

```bash
docker compose build
NIFTI_HOST_PORT=9000 docker compose up -d app
```

## 验证（verify 一次性服务）

```bash
docker compose up --build --exit-code-from verify verify
```

`verify` 等待 `app` 健康检查通过后执行三个阶段，退出码按位汇总：

| 位 | 值 | 阶段 |
| --- | --- | --- |
| bit0 | 1 | 代码测试（`tests/` 单元 + 集成测试） |
| bit1 | 2 | 镜像构建校验（构建清单、运行时版本、模块导入、采样自检） |
| bit2 | 4 | API 冒烟（大/小端 × sform/qform × int16/float32 的 sample 与 sphere-stats 双矩阵 + 结构错误、逐点与逐区错误用例） |

退出码 0 表示全部通过。本地复现（stage 2 需要镜像构建清单
`image-manifest.json`，仅在 Dockerfile 构建时生成）：

```bash
python -m unittest discover -s tests -t .          # 仅代码测试
VERIFY_BASE_URL=http://127.0.0.1:8000 python -m verify.verify
```

## 目录结构

```
app/            服务实现（server: HTTP/multipart；nifti: 头部解析校验；sampling: 三线性插值；sphere: 闭球统计）
tests/          单元与集成测试（stdlib unittest）
verify/         一次性校验服务 + NIfTI 样本生成器 + multipart 客户端
Dockerfile      单阶段镜像（python:3.11-slim，无外部依赖）
docker-compose.yml  app（健康检查、可配置宿主机端口）+ verify（一次性）
```
