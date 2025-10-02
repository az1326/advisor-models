import numpy as np
import pandas as pd
from pprint import pprint
from typing import List, Dict, Any

bag_list_example = [{"name": "luggage box", "size": [33, 18, 12], "weight": 67}]

complementary_first = [
    "China",
    "Hong Kong",
    "Japan",
    "South Korea",
    "India",
    "Qatar",
    "Haiti",
    "Cuba",
    "Panama",
    "Colombia",
    "Ecuador",
    "Peru",
    "South America",
    "Israel",
]


def invert_order(a, order):
    return [a[i] for i in np.argsort(order)]


def load_checking_fee():
    check_base = []
    for bag_num in range(1, 5):
        us_departure = pd.read_csv(f"fee_tables/bag_{bag_num}/0.csv", index_col=0)
        us_arrival = pd.read_csv(f"fee_tables/bag_{bag_num}/1.csv", index_col=0)
        check_base.append({0: us_departure, 1: us_arrival})
    return check_base


check_base_tables = load_checking_fee()


def compute_answer(
    base_price: int,
    direction: int,
    routine: str,
    customer_class: str,
    bag_list: List[Dict[str, Any]],
    check_base_tables: List[Dict[int, pd.DataFrame]],
    override: dict = {},
):
    extra, info_dict = compute_check_cost(
        bag_list[1:], direction, routine, customer_class, check_base_tables, override
    )
    extra = override.get("check_total", extra)
    total_cost = base_price + extra
    info_dict.update(
        {
            "customer_class": customer_class,
            "ticket_price": base_price,
            "place_of_departure": routine if direction == 1 else "U.S.",
            "place_of_arrival": routine if direction == 0 else "U.S.",
            "routine": routine,
            "total_cost": total_cost,
            "bag_list": bag_list[1:],
        }
    )
    return total_cost, info_dict


def compute_check_cost(
    bag_list: List[Dict[str, float]],
    direction: int,
    routine: str,
    customer_class: str,
    check_base_tables: List[Dict[int, pd.DataFrame]],
    override: dict = {},
):
    if "check_base" not in override:
        oversize_cost = [compute_oversize(b, routine) for b in bag_list]
        overweight_cost_if_comp = [
            compute_overweight(b, routine, customer_class, True) for b in bag_list
        ]

        overweight_cost_if_not_comp = [
            compute_overweight(b, routine, customer_class, False) for b in bag_list
        ]

        violation_cost_if_comp = np.maximum(oversize_cost, overweight_cost_if_comp)
        violation_cost_if_not_comp = np.maximum(
            oversize_cost, overweight_cost_if_not_comp
        )

        complementary_gain = violation_cost_if_not_comp - violation_cost_if_comp
        order = np.argsort(-complementary_gain)
        bag_list = [bag_list[i] for i in order]
        check_base = compute_base(
            bag_list, direction, routine, customer_class, check_base_tables
        )
    else:
        check_base = override["check_base"]
        order = np.arange(len(bag_list))

    complementary = [(x == 0) for x in check_base]
    oversize_cost = [compute_oversize(b, routine) for b in bag_list]
    overweight_cost = [
        compute_overweight(b, routine, customer_class, c)
        for b, c in zip(bag_list, complementary)
    ]
    violation_cost = np.maximum(oversize_cost, overweight_cost).sum()
    total_check_cost = np.sum(check_base) + violation_cost
    info_dict = {
        "overweight": invert_order(overweight_cost, order),
        "oversize": invert_order(oversize_cost, order),
        "base": invert_order(check_base, order),
    }
    return total_check_cost, info_dict


def compute_base(
    bag_list: List[Dict[str, float]],
    direction: int,
    routine: str,
    customer_class: str,
    check_base_tables: List[Dict[int, pd.DataFrame]],
):
    check_base = []
    for bag_id, _ in enumerate(bag_list):
        bag_id = min(3, bag_id)
        check_base.append(check_base_tables[bag_id][direction][customer_class][routine])
    return check_base


def compute_oversize(bag: Dict[str, float], routine: str):
    if np.sum(bag["size"]) <= 62:
        return 0
    if np.sum(bag["size"]) > 62 and np.sum(bag["size"]) <= 65:
        return 30
    if routine in [
        "Panama",
        "South America",
        "Peru",
        "Colombia",
        "Ecuador",
        "Europe",
        "Israel",
        "Qatar",
    ]:
        return 150
    return 200


def compute_overweight(
    bag: Dict[str, float], routine: str, customer_class: str, complementary: bool
):
    if routine in ["Australia", "New Zealand"]:
        if complementary:
            if bag["weight"] <= 70:
                return 0
            return 200
        if bag["weight"] <= 50:
            return 0
        if bag["weight"] > 50 and bag["weight"] <= 53:
            return 30
        if bag["weight"] > 53 and bag["weight"] <= 70:
            if routine == "Cuba":
                return 200
            return 100
        if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        return 200
    if complementary and customer_class in ["Business", "First"]:
        if bag["weight"] <= 70:
            return 0
        elif routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        else:
            return 200
    if bag["weight"] <= 50:
        return 0
    if bag["weight"] > 50 and bag["weight"] <= 53:
        return 30
    if bag["weight"] > 53 and bag["weight"] <= 70:
        if routine == "Cuba":
            return 200
        return 100
    if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
        return 450
    return 200


if __name__ == "__main__":
    from pprint import pprint
    from gen_questions import gen_question

    prompt, info_dict = gen_question("Jim", 5, 3)
    pprint(info_dict)
    check_base_tables = load_checking_fee()
    fee = compute_answer(**info_dict, check_base_tables=check_base_tables)
    print(fee)
