#!/usr/bin/env bash
# Build test data, run the notebook with and without streams, and check both passes independently.
# Usage: bash tests/run_all.sh [WORK_DIR]       (run from the handoff folder)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NB="$HERE/../wetland_drainage_areas.ipynb"
W="${1:-$HERE/_work}"

python "$HERE/make_test_data.py" "$W/data"

for case in nostreams streams; do
  streams=None
  extra=()
  if [ "$case" = streams ]; then streams="r'$W/data/streams.shp'"; extra=(--streams "$W/data/streams.shp"); fi
  echo "=== $case ==="
  python "$HERE/run_notebook.py" "$NB" \
    "DEM_PATH=r'$W/data/dem.tif'" "WETLANDS_PATH=r'$W/data/wetlands.gpkg'" \
    "OUT_DIR=r'$W/$case'" "PREFIX='test'" "STREAMS_PATH=$streams" "WRITE_POLYGONS=True"
  python "$HERE/check_pass1.py" "$W/data/dem.tif" "$W/data/wetlands.gpkg" "$W/$case/test" "${extra[@]}"
  python "$HERE/check_pass2.py" "$W/data/dem.tif" "$W/$case/test"
done
python "$HERE/check_cases.py" "$W/data" "$W/nostreams/test"

echo "=== rounded DEM (exact ties) ==="
python "$HERE/run_notebook.py" "$NB" \
  "DEM_PATH=r'$W/data/dem_rounded.tif'" "WETLANDS_PATH=r'$W/data/wetlands.gpkg'" \
  "OUT_DIR=r'$W/rounded'" "PREFIX='test'" "STREAMS_PATH=r'$W/data/streams.shp'"
python "$HERE/check_pass1.py" "$W/data/dem_rounded.tif" "$W/data/wetlands.gpkg" "$W/rounded/test" --streams "$W/data/streams.shp"
python "$HERE/check_pass2.py" "$W/data/dem_rounded.tif" "$W/rounded/test"
echo "ALL CHECKS PASSED"
