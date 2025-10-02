import numpy as np

from enum import Enum
from typing import Any
from pydantic import BaseModel

from compute_answer import compute_answer, load_checking_fee, compute_overweight
from gen_questions import international_city_collection

rule_list = [
    "1 base check price matching",
    "2 base check price matching",
    "3 base check price matching",
    "4+ base check price matching",
    "oversize fee matching",
    "overweight fee matching",
    "complementary overweight",
    "maximum violation fee",
    "main plus extra complementary",
    "overall fee aggregation",
]


class CheckType(str, Enum):
    carry = "Carry-On Bag"
    free = "Complementary"
    charged = "Charged"


class SizeInterval(str, Enum):
    le_62in = "Within 62 in / 158 cm"
    g_62in_le_65in = "Over 62 in / 158 cm – 65 in / 165 cm"
    g_65in_le_115in = "Over 65 in / 165 cm – 115 in / 292 cm"


class WeightInterval(str, Enum):
    le_50lbs = "Within 50 lbs / 23 kgs"
    g_50lbs_le_53lbs = "Over 50 lbs / 23 kgs – 53 lbs / 24 kgs"
    g_53lbs_le_70lbs = "Over 53 lbs / 24 kgs – 70 lbs / 32 kgs"
    g_70lbs_le_100lbs = "Over 70 lbs / 32 kgs – 100 lbs / 45 kgs"


class BagCost(BaseModel):
    size: int
    weight: int
    base_check_fee: int
    oversize_fee: int
    overweight_fee: int
    total_fee: int


class PassengerClass(str, Enum):
    be = "Basic Economy"
    main = "Main Cabin"
    mp = "Main Plus"
    pe = "Premium Economy"
    business = "Business"
    first = "First"


class Response(BaseModel):
    passenger_class: str
    place_of_departure: str
    place_of_arrival: str
    ticket_price: int
    checked_bags: list[BagCost]
    total_cost: int

    def set_reference_overweight(self, info_dict):
        # NOTE: Can't set `extra=True` for `BagCost` due to GPT-4o limitation
        #       Using __dict__ to implement is not elegant
        for bag in self.checked_bags:
            bag.__dict__["reference_overweight"] = compute_overweight(
                {"weight": bag.weight},
                info_dict["routine"],
                info_dict["customer_class"],
                False,
            )


def build_reference(info_dict: dict[str, Any]):
    checked_bags = []

    for bag, ow, os, bs in zip(
        info_dict["bag_list"],
        info_dict["overweight"],
        info_dict["oversize"],
        info_dict["base"],
    ):
        size = np.sum(bag["size"])
        weight = bag["weight"]

        checked_bags.append(
            BagCost(
                size=size,
                weight=weight,
                base_check_fee=bs,
                oversize_fee=os,
                overweight_fee=ow,
                total_fee=bs + max(os, ow),
            )
        )

    reference = Response(
        passenger_class=info_dict["customer_class"],
        place_of_departure=info_dict["place_of_departure"],
        place_of_arrival=info_dict["place_of_arrival"],
        ticket_price=info_dict["ticket_price"],
        checked_bags=checked_bags,
        total_cost=info_dict["total_cost"],
    )
    reference.set_reference_overweight(info_dict)
    return reference


# True if info in response is corrct, otherwise False
def check_basic_travel_info(response: Response, reference: Response):
    if response.passenger_class != reference.passenger_class:
        return True

    departure = international_city_collection.get(response.place_of_departure, "U.S.")
    arrival = international_city_collection.get(response.place_of_arrival, "U.S.")
    if (
        departure != reference.place_of_departure
        or arrival != reference.place_of_arrival
    ):
        return True

    if response.ticket_price != reference.ticket_price:
        return True

    return False


# Decide if price match error or main plus error
def check_base_bag_payment(response: Response, reference: Response):
    examine_list = []
    pred_num_free = np.sum([bag.base_check_fee == 0 for bag in response.checked_bags])
    ref_num_free = np.sum([bag.base_check_fee == 0 for bag in reference.checked_bags])
    if reference.passenger_class == "Main Plus" and (pred_num_free == ref_num_free):
        examine_list.append("Correct: main plus extra complementary")
    elif reference.passenger_class == "Main Plus" and (
        pred_num_free == ref_num_free - 1
    ):
        examine_list.append("Missing: main plus extra complementary")
    elif reference.passenger_class == "Main Plus":
        examine_list.append("Error: main plus extra complementary")

    pred_fees = np.sort([bag.base_check_fee for bag in response.checked_bags])
    ref_fees = np.sort([bag.base_check_fee for bag in reference.checked_bags])
    for idx, (pred_bag, ref_bag) in enumerate(zip(pred_fees, ref_fees)):
        idx = str(idx + 1) if idx < 3 else "4+"
        if pred_bag != ref_bag:
            if pred_bag == 0:
                examine_list.append(f"Missing: {idx} base check price matching")
            else:
                examine_list.append(f"Error: {idx} base check price matching")
        else:
            examine_list.append(f"Correct: {idx} base check price matching")
    return examine_list


# Categorize the error in overweight and oversize fee calculation
def check_overweight_and_oversize(response: Response, reference: Response):
    examine_list = []
    for pred_bag, ref_bag in zip(response.checked_bags, reference.checked_bags):
        if pred_bag.size != ref_bag.size:
            examine_list.append("Error: size categorization")
        if pred_bag.oversize_fee != ref_bag.oversize_fee:
            if pred_bag.oversize_fee == 0:
                examine_list.append("Missing: oversize fee matching")
            else:
                examine_list.append("Error: oversize fee matching")
        else:
            examine_list.append("Correct: oversize fee matching")

        if pred_bag.weight != ref_bag.weight:
            examine_list.append("Error: weight categorization")
        if pred_bag.overweight_fee != ref_bag.overweight_fee:
            if (
                ref_bag.weight > 50
                and ref_bag.weight <= 70
                and ref_bag.overweight_fee == 0
                and pred_bag.overweight_fee == ref_bag.reference_overweight
            ):
                examine_list.append("Missing: complementary overweight")
            elif pred_bag.overweight_fee == 0:
                examine_list.append("Missing: overweight fee matching")
            else:
                examine_list.append("Error: overweight fee matching")
        else:
            if (
                ref_bag.weight > 50
                and ref_bag.weight <= 70
                and ref_bag.overweight_fee == 0
            ):
                examine_list.append("Correct: complementary overweight")
            examine_list.append("Correct: overweight fee matching")

        if (
            pred_bag.total_fee != ref_bag.total_fee
            and ref_bag.overweight_fee > 0
            and ref_bag.oversize_fee > 0
        ):
            if (
                pred_bag.total_fee
                == pred_bag.base_check_fee
                + pred_bag.oversize_fee
                + pred_bag.overweight_fee
            ):
                examine_list.append("Missing: maximum violation fee")
            elif pred_bag.total_fee == pred_bag.base_check_fee + max(
                pred_bag.oversize_fee, pred_bag.overweight_fee
            ):
                examine_list.append("Correct: maximum violation fee")

    return examine_list


def check_overall(response: Response, reference: Response):
    if response.total_cost != reference.total_cost:
        return "Error: overall fee aggregation"
    return "Correct: overall fee aggregation"


# TODO: Analyze the source of error in LLMs' `response`
def error_analysis(response: Response, question_info_dict: dict):
    def recalc_answer(question_info_dict, check_base_tables, override):
        _, info_dict_for_parse = compute_answer(
            **question_info_dict, check_base_tables=check_base_tables, override=override
        )
        reference = build_reference(info_dict_for_parse)
        return reference

    # NOTE: Initialize reference answer
    check_base_tables = load_checking_fee()
    reference = recalc_answer(question_info_dict, check_base_tables, {})
    examine_list = []

    base_check_fee_judgement = check_base_bag_payment(response, reference)
    if len(base_check_fee_judgement):
        examine_list.extend(base_check_fee_judgement)
        override_check_base = [bag.base_check_fee for bag in response.checked_bags]

    # Override the base check fee if LLMs have made a mistake
    override = (
        {"check_base": override_check_base} if len(base_check_fee_judgement) else {}
    )
    reference = recalc_answer(question_info_dict, check_base_tables, override)

    overweight_oversize_fee_errors = check_overweight_and_oversize(response, reference)
    examine_list.extend(overweight_oversize_fee_errors)

    override.update(
        check_total=np.sum([bag.total_fee for bag in response.checked_bags])
    )
    reference = recalc_answer(question_info_dict, check_base_tables, override)

    overall_calculation_judgement = check_overall(response, reference)
    if overall_calculation_judgement is not None:
        examine_list.append(overall_calculation_judgement)

    return examine_list
