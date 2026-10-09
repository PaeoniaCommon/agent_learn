"""A stand-in for the real RPA module, for trying the agent locally."""

import re

import pandas as pd

RPA_VIEWS = ["sales_daily", "sales_weekly", "stock_level"]

_PARAMS = {
    "sales_daily": {"mandatory": ["region", "date_from", "date_to"], "optional": ["currency", "store_id"],
                    "default": {"currency": "GBP"}},
    "sales_weekly": {"mandatory": ["region", "date_from", "date_to"], "optional": ["currency"],
                     "default": {"currency": "GBP"}},
    "stock_level": {"mandatory": ["region", "as_of_date"], "optional": [], "default": {}},
}
_VALID = {"region": ["UK", "IE", "FR"], "currency": ["GBP", "EUR"]}


def validate_view(view):
    return view in RPA_VIEWS


def get_view_params(view):
    return _PARAMS[view]


def valid_params(param):
    return _VALID.get(param, [])


def get_data(view, param_dict):
    for p in ("date_from", "date_to", "as_of_date"):
        if p in param_dict and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(param_dict[p])):
            raise ValueError(f"{p}: unparseable date, expected YYYY-MM-DD")
    if "store_id" in param_dict and not str(param_dict["store_id"]).isdigit():
        raise ValueError("store_id must be numeric")
    return pd.DataFrame({"store_id": [101, 102], "sku": ["A-22", "B-1"], "units": [4, 2], "revenue": [19.96, 5.0]})
