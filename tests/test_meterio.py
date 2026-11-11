import numpy as np
import pandas as pd
import pytest
from conftest import TZ, rows_from, synthetic

from interval_vee.meterio import ingest, read_register, read_rows, to_utc, write_register


def test_offset_timestamps_parse_to_utc():
    utc, nonexist = to_utc(pd.Series(["2009-10-25T02:15:00+02:00", "2009-10-25T02:15:00+01:00", "2009-10-25T00:00:00Z"]), TZ)
    assert [str(t) for t in utc] == ["2009-10-25 00:15:00+00:00", "2009-10-25 01:15:00+00:00", "2009-10-25 00:00:00+00:00"]
    assert not nonexist.any()


def test_naive_autumn_hour_first_is_summer_second_is_winter():
    utc, _ = to_utc(pd.Series(["2009-10-25 02:00", "2009-10-25 02:00", "2009-10-25 03:00"]), TZ)
    assert [t.strftime("%H:%M") for t in utc] == ["00:00", "01:00", "02:00"]


def test_naive_spring_hour_that_does_not_exist_is_rejected():
    rows = pd.DataFrame({"timestamp": ["2010-03-28 01:45", "2010-03-28 02:15", "2010-03-28 03:00"], "kwh": ["0.1", "0.2", "0.3"]})
    ing = ingest(rows, TZ)
    assert ing.issues["nonexistent_local_time"] == 1
    assert ing.issues["accepted"] == 2


def test_dst_days_have_92_and_100_slots_after_round_trip():
    s = synthetic(days=14, start="2009-10-19")
    ing = ingest(rows_from(s), TZ, grid=s.index)
    assert ing.issues["accepted"] == len(s)
    assert (ing.counts == 1).all()
    np.testing.assert_allclose(ing.value[np.argsort(ing.slot)], s.to_numpy())
    local = s.index.tz_convert(TZ).tz_localize(None).normalize()
    assert pd.Series(1, index=local).groupby(level=0).size().loc["2009-10-25"] == 100
    s2 = synthetic(days=3, start="2010-03-27")
    local2 = s2.index.tz_convert(TZ).tz_localize(None).normalize()
    assert pd.Series(1, index=local2).groupby(level=0).size().loc["2010-03-28"] == 92


def test_off_grid_unparseable_and_outside_rows_are_counted():
    s = synthetic(days=1)
    rows = rows_from(s)
    extra = pd.DataFrame({"timestamp": ["2009-10-05 10:07", "2009-10-05 10:15", "2030-01-01 00:00", "garbage"],
                          "kwh": ["0.1", "abc", "0.1", "0.1"]})
    ing = ingest(pd.concat([rows, extra], ignore_index=True), TZ, grid=s.index)
    assert ing.issues["off_grid"] == 1
    assert ing.issues["unparseable_value"] == 1
    assert ing.issues["outside_period"] == 1
    assert ing.issues["bad_timestamp"] == 1
    assert ing.issues["accepted"] == len(s)


def test_duplicate_rows_are_kept_with_file_order():
    s = synthetic(days=1)
    rows = rows_from(s)
    rows = pd.concat([rows, rows.iloc[[10]].assign(kwh="9.999")], ignore_index=True)
    ing = ingest(rows, TZ, grid=s.index)
    assert ing.counts[10] == 2
    sel = ing.slot == 10
    assert list(ing.value[sel][np.argsort(ing.order[sel])]) == [s.iloc[10], 9.999]


def test_read_rows_needs_timestamp_and_kwh(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("time,value\n2009-10-01 00:00,0.1\n")
    with pytest.raises(ValueError):
        read_rows(p)
    p.write_text("Timestamp,KWH\n2009-10-01 00:00,0.1\n")
    assert list(read_rows(p).columns) == ["timestamp", "kwh"]


def test_register_round_trip(tmp_path):
    reg = pd.Series([12.345, 7.0], index=[pd.Timestamp("2009-10-01").date(), pd.Timestamp("2009-10-02").date()])
    write_register(reg, tmp_path / "r.csv")
    back = read_register(tmp_path / "r.csv")
    assert list(back.round(2)) == [12.35, 7.0] or list(back.round(2)) == [12.34, 7.0]
    assert back.index[0] == reg.index[0]
