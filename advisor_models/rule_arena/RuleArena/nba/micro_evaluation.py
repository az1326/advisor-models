import os
import openai
from pydantic import BaseModel, Field

system_prompt = "You are a helpful NBA team consultant and an expert at structured data extraction. You are given some relevant rules in NBA Collective Bargaining Agreement (CBA), a scenario where one or more teams plan to implement one or more signing or trade operations, and an analysis to verify whether these operations satisfy the rules. You should determine whether each rule is involved in the analysis for each operation and output it in the given structure."


def gpt(inputs):
    api_key = os.environ["OPENAI_API_KEY"]
    engine = openai.OpenAI(api_key=api_key)
    response = engine.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inputs},
        ],
        response_format=Response,
        temperature=0.0,
    )
    return response.choices[0].message.parsed


class RuleExtraction(BaseModel):
    # contract length
    contract_length_at_most_4_year_except_qualifying_veteran_free_agent_5_year: bool = Field(
        description="A Player Contract may cover, in the aggregate, up to but no more than four (4) Seasons from the date such Contract is signed; provided, however, that a Player Contract between a Qualifying Veteran Free Agent and his Prior Team may cover, in the aggregate, up to but no more than five (5) Seasons from the date such Contract is signed."
    )
    contract_length_at_most_2_year_bi_annual_exception: bool = Field(
        description="A Player Contract signed using Bi-annual Exception may not exceed two (2) Seasons in length."
    )
    contract_length_at_most_4_year_non_taxpayer_mid_level_exception: bool = Field(
        description="A Player Contract signed using Non-Taxpayer Mid-Level Exception may not exceed two (4) Seasons in length."
    )
    contract_length_at_most_2_year_taxpayer_mid_level_exception: bool = Field(
        description="A Player Contract signed using Taxpayer Mid-Level Exception may not exceed two (2) Seasons in length."
    )
    contract_length_at_most_3_year_mid_level_exception_for_room_team: bool = Field(
        description="A Player Contract signed using Mid-Level Exception for Room Teams may not exceed two (3) Seasons in length."
    )
    contract_length_at_most_2_year_minimum_player_salary_exception: bool = Field(
        description="A Player Contract signed using Minimum Player Salary Exception may not exceed two (2) Seasons in length."
    )

    # basic rules
    salary_cap_no_exceed_without_exception: bool = Field(
        description="A Team’s Team Salary may not exceed the Salary Cap at any time unless the Team is using one of the Exceptions."
    )
    maximum_salary_for_player_less_than_7_year_service: bool = Field(
        description="For any player who has completed fewer than seven (7) Years of Service, his Player Contract may not provide for a Salary in the first Season that exceeds twenty-five percent (25%) of the Salary Cap."
    )
    maximum_salary_for_player_7_to_9_year_service: bool = Field(
        description="For any player who has completed at least seven (7) but fewer than ten (10) Years of Service, his Player Contract may not provide for a Salary in the first Season covered by the Contract that exceeds thirty percent (30%) of the Salary Cap."
    )
    maximum_salary_for_player_10_or_more_year_service: bool = Field(
        description="For any player who has completed ten (10) or more Years of Service, his Player Contract may not provide for a Salary in the first Season covered by the Contract that exceeds thirty-five percent (35%) of the Salary Cap."
    )
    higher_max_criterion_for_5th_year_eligible_player: bool = Field(
        description="A player who has four (4) Years of Service covered by his Player Contract (“5th Year Eligible Players”) shall be eligible to receive from his Prior Team up to thirty percent (30%) of the Salary Cap in effect at the time the Contract is executed if the player has met at least one of the “Higher Max Criteria”."
    )
    salary_increase_and_decrease_ratio_except_qualiyfing_or_early_qualifying_veteran_free_agent: bool = Field(
        description="For all Player Contracts other than Contracts between Qualifying Veteran Free Agents or Early Qualifying Veteran Free Agents and their Prior Team: For each Salary Cap Year covered by a Player Contract after the first Salary Cap Year, the player’s Salary may increase or decrease in relation to the previous Salary Cap Year’s Salary by no more than five percent (5%) of the Salary for the first Salary Cap Year covered by the Contract."
    )
    salary_increase_and_decrease_ratio_for_qualiyfing_or_early_qualifying_veteran_free_agent: bool = Field(
        description="For all Player Contracts between Qualifying Veteran Free Agents or Early Qualifying Veteran Free Agents and their Prior Team (except any such Contracts signed pursuant to a Bi-Annual Exception, any kind of Mid-Level Exception, and sign-and-trade): For each Salary Cap Year covered by a Player Contract after the first Salary Cap Year, the player’s Salary may increase or decrease in relation to the previous Salary Cap Year’s Salary by no more than eight percent (8%) of the Salary for the first Salary Cap Year covered by the Contract."
    )

    # 38 year old provision
    defer_compensation_38_year_old: bool = Field(
        description="Except a Qualifying Veteran Free Agent who is age 35 or 36, the aggregate Salaries in an Over 38 Contract (covers four (4) or more Seasons, including one (1) or more Seasons commencing after such player will reach or has reached age thirty-eight (38)) for Salary Cap Years commencing with the fourth Salary Cap Year of such Over 38 Contract or the first Salary Cap Year that covers a Season that follows the player’s 38th birthday, whichever is later, shall be attributed to the prior Salary Cap Years pro rata on the basis of the Salaries for such prior Salary Cap Years."
    )
    defer_compensation_qualifying_veteran_free_agent_38_year_old: bool = Field(
        description="If a Qualifying Veteran Free Agent who is age 35 or 36 enters into an Over 38 Contract (covers four (4) or more Seasons, including one (1) or more Seasons commencing after such player will reach or has reached age thirty-eight (38)) with his Prior Team covering five (5) Seasons, the Salary in such Over 38 Contract for the fifth Salary Cap Year shall be attributed to the prior Salary Cap Years pro rata on the basis of the Salaries for such prior Salary Cap Years."
    )

    # apron level as hard cap rules
    bi_annual_exception_hard_cap_first_apron_level: bool = Field(
        description="A Team may not sign or acquire a player using the Bi-annual Exception if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the First Apron Level."
    )
    non_taxpayer_mid_level_exception_hard_cap_first_apron_level: bool = Field(
        description="A Team may not sign or acquire a player using the Non-Taxpayer Mid-Level Exception if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the First Apron Level."
    )
    sign_and_trade_hard_cap_first_apron_level: bool = Field(
        description="A Team may not acquire a player pursuant to a Contract entered into in accordance with sign-and-trade if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the First Apron Level."
    )
    expanded_traded_player_exception_hard_cap_first_apron_level: bool = Field(
        description="A Team may not acquire a player using an Expanded Traded Player Exception if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the First Apron Level."
    )
    aggregated_traded_player_exception_hard_cap_second_apron_level: bool = Field(
        description="A Team may not acquire a player using an Aggregated Standard Traded Player Exception if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the Second Apron Level."
    )
    cash_in_trade_hard_cap_second_apron_level: bool = Field(
        description="A Team may not pay cash to another Team in connection with a trade if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the Second Apron Level."
    )
    sign_and_trade_assigner_traded_player_exception_hard_cap_second_apron_level: bool = Field(
        description="A Team may not acquires a player using a Traded Player Exception, which Traded Player Exception is in respect of a Player Contract signed and traded, if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the Second Apron Level."
    )
    taxpayer_mid_level_exception_hard_cap_second_apron_level: bool = Field(
        description="A Team may not sign a player using the Taxpayer Mid-Level Salary Exception if, immediately following such transaction, the Team’s Team Salary for such Salary Cap Year would exceed the Second Apron Level."
    )
    traded_player_exception_250k_reduced_first_apron_level: bool = Field(
        description="If a Team’s post-assignment Team Salary would exceed the First Apron Level, then the $250,000 allowance referenced in each Traded Player Exception shall be reduced to $0."
    )

    # exceptions

    # bird rights
    qualifying_veteran_free_agent_exception: bool = Field(
        description="A Qualifying Veteran Free Agent may have a Salary Cap Excetion, according to which the player may enter into a new Player Contract with his Prior Team."
    )
    early_qualifying_veteran_free_agent_exception: bool = Field(
        description="A Early Qualifying Veteran Free Agent may have a Salary Cap Excetion, according to which the player may enter into a new Player Contract with his Prior Team."
    )
    non_qualifying_veteran_free_agent_exception: bool = Field(
        description="A Non-Qualifying Veteran Free Agent may have a Salary Cap Excetion, according to which the player may enter into a new Player Contract with his Prior Team."
    )
    salary_space_consumption_qualifying_veteran_free_agent: bool = Field(
        description="For purposes of computing Team Salary, a Qualifying Veteran Free Agent, other than who follows the second Option Year of his Rookie Scale Contract, will be included at no less than one hundred fifty percent (150%) of his prior Salary."
    )
    salary_space_consumption_early_qualifying_veteran_free_agent: bool = Field(
        description="For purposes of computing Team Salary, an Early Qualifying Veteran Free Agent will be included at one hundred thirty percent (130%) of his prior Salary."
    )
    salary_space_consumption_non_qualifying_veteran_free_agent: bool = Field(
        description="For purposes of computing Team Salary, a Non-Qualifying Veteran Free Agent will be included at one hundred twenty percent (120%) of his prior Salary."
    )
    salary_space_consumption_standard_traded_player_exception: bool = Field(
        description="For purposes of computing Team Salary, the Standard Traded Player Exception will be included at one hundred percent (100%) of its amount."
    )

    # bi-annual exception
    bi_annual_exception: bool = Field(
        description="A Team may use the Bi-annual Exception during a Salary Cap Year to sign and/or acquire by assignment one (1) or more Player Contracts during each Salary Cap Year (i) if the Team has not used the Mid-Level Salary Exception for Room Teams in that same Salary Cap Year, and (ii) not in any two (2) consecutive Salary Cap Years."
    )

    # mid level exceptions
    non_taxpayer_mid_level_exception: bool = Field(
        description="A Team may use the Non-Taxpayer Mid-Level Salary Exception to sign and/or acquire by assignment one (1) or more Player Contracts during each Salary Cap Year. If a Veteran Free Agent with one (1) or two (2) Years of Service receives an Offer Sheet, the player’s Prior Team may use the Non-Taxpayer Mid-Level Salary Exception to match the Offer Sheet."
    )
    taxpayer_mid_level_exception: bool = Field(
        description="A Team may use the Taxpayer Mid-Level Salary Exception to sign one (1) or more Player Contracts during each Salary Cap Year not to exceed two (2) Seasons in length, that, in the aggregate, provide for Salaries up to about $5 million."
    )
    mid_level_exception_for_room_team: bool = Field(
        description="In the event (i) a Team’s Team Salary at any time during a Salary Cap Year is below the Salary Cap for such Salary Cap Year, and (ii) at the time the Team proposes to use the Mid-Level Salary Exception for Room Teams, the Team has not already used either the Bi-annual Exception, the Non-Taxpayer Mid-Level Salary Exception, or the Taxpayer Mid-Level Salary Exception in that same Salary Cap Year, then the Team may at such time use the Mid-Level Salary Exception for Room Teams to sign and/or acquire by assignment one (1) or more Player Contracts."
    )
    minimum_player_salary_exception: bool = Field(
        description="A Team may sign a player to, or acquire by assignment, a Player Contract, not to exceed two (2) Seasons in length, that provides for a Salary for the first Season equal to the Minimum Player Salary applicable to that player."
    )

    # traded player exceptions
    standard_traded_player_exception: bool = Field(
        description="A Team may use the “Standard Traded Player Exception” to replace one (1) Traded Player with one (1) or more Replacement Players whose Player Contracts are acquired simultaneously or non-simultaneously and whose post-assignment Salaries for the Salary Cap Year in which the Replacement Player(s) are acquired, in the aggregate, are no more than an amount equal to one hundred percent (100%) of the pre-trade Salary of the Traded Player, plus $250,000."
    )
    aggregated_standard_traded_player_exception: bool = Field(
        description="A Team may use the “Aggregated Standard Traded Player Exception” to replace two (2) or more Traded Players with one (1) or more Replacement Players whose Player Contracts are acquired simultaneously and whose post-trade Salaries for the then-current Salary Cap Year, in the aggregate, are no more than an amount equal to one hundred percent (100%) of the aggregated pre-trade Salaries of the Traded Players, plus $250,000."
    )
    expanded_traded_player_exception: bool = Field(
        description="A Team may use the “Expanded Traded Player Exception” to replace one (1) or more Traded Players with one (1) or more Replacement Players whose Player Contracts are acquired simultaneously and whose post-trade Salaries for the then-current Salary Cap Year, in the aggregate, are no more than an amount not less than one hundred twenty-five percent (125%) of the aggregated pre-trade Salaries of the Traded Player(s), plus $250,000."
    )
    traded_player_exception_for_room_team: bool = Field(
        description="A Team with a Team Salary below the Salary Cap may acquire one (1) or more players by assignment whose post-assignment Salaries, in the aggregate, are no more than an amount equal to the Team’s room under the Salary Cap plus $250,000."
    )
    traded_player_exception_only_one_minimum_traded_player_under_conditions: bool = Field(
        description="Other than during the period beginning on December 15 of a Salary Cap Year through the NBA trade deadline of such Salary Cap Year, if a Team is aggregating the Contracts of three (3) or more Traded Players in a trade and the number of Replacement Players that the Team is acquiring in respect of such Traded Players is less than the number of such Traded Players, then no more than one (1) of such Traded Players may be a Minimum Salary Player."
    )

    # trade rules
    pay_or_receive_cash_maximum_in_a_year: bool = Field(
        description="A Team shall be permitted to pay or receive in connection with one (1) or more trades during a Salary Cap Year up to an aggregate amount equal to 5.15% of the Salary Cap for such Salary Cap Year in cash across all such trades."
    )
    rookie_or_two_way_contract_cannot_be_traded_within_30_days: bool = Field(
        description="No Draft Rookie who signs a Standard NBA Contract or player who signs a Two-Way Contract may be traded before thirty (30) days following the date on which the Contract is signed."
    )
    free_agent_sign_contract_cannot_be_traded_within_3_month_or_before_dec_15: bool = Field(
        description="No player who signs a Standard NBA Contract as a Free Agent (or who signs a Standard NBA Contract while under a Two-Way Contract) may be traded before the later of (A) three (3) months following the date on which such Contract was signed or (B) the December 15 of the Salary Cap Year in which such Contract was signed."
    )
    qualifying_or_early_qualifying_free_agent_sign_contract_cannot_be_traded_within_3_month_or_before_jan_15: bool = Field(
        description="Any player who signs a Standard NBA Contract with his prior Team meeting the following criteria may not be traded before the later of (x) three (3) months following the date on which such Contract was signed or (y) the January 15 of the Salary Cap Year in which such Contract was signed: the Team Salary of the player’s Team is above the Salary Cap immediately following the Contract signing and the player is a Qualifying Veteran Free Agent or Early Qualifying Veteran Free Agent who enters into a new Player Contract with his prior Team that provides for a Salary for the first Season of such new Contract greater than one hundred twenty percent (120%) of the Salary for the last Season of the player’s immediately prior Contract."
    )

    # sign-and-trade rules
    sign_and_trade_3_to_4_year: bool = Field(
        description="In sign-and-trade scenario, a Veteran Free Agent and his Prior Team may enter into a Player Contract to be traded for at least three (3) Seasons (excluding any Option Year) but no more than four (4) Seasons in length."
    )
    sign_and_trade_not_with_mid_level_exception: bool = Field(
        description="In sign-and-trade scenario, a Veteran Free Agent and his Prior Team may not enter into a Player Contract to be traded pursuant to the Mid-Level Salary Cap Exceptions."
    )
    sign_and_trade_no_higher_than_25_percent_for_higher_max_5th_year_eligible_player: bool = Field(
        description="A Veteran Free Agent and his Prior Team may enter into a Player Contract pursuant to an agreement between the Prior Team and another Team concerning the signing and subsequent trade of such Contract only if with respect to any 5th Year Eligible Player who met one of the Higher Max Criteria, the Contract may not provide the player with Salary in excess of twenty-five percent (25%) of the Salary Cap."
    )
    sign_and_trade_assignee_team_has_room: bool = Field(
        description="A Veteran Free Agent and his Prior Team may enter into a Player Contract pursuant to an agreement between the Prior Team and another Team concerning the signing and subsequent trade of such Contract only if the acquiring Team has Room for the player’s Salary provided for in the first Season of the Contract."
    )
    sign_and_trade_qualifying_free_agent_half_salary_for_traded_player_exception: bool = Field(
        description="In sign-and-trade scenario, to compute the amount of the Traded Player Exception, the Salary of the Qualifying Veteran Free Agent or Early Qualifying Veteran Free Agent should be deemed reduced in some cases."
    )

    # restricted free agent rules (Arenas provision)
    offer_sheet_for_1_or_2_year_service_player_no_more_than_mid_level_in_first_2_year: bool = Field(
        description="No Offer Sheet may provide for Salary in the first Salary Cap Year totaling more than the amount of the Non-Taxpayer Mid-Level Salary Exception for such Salary Cap Year."
    )
    offer_sheet_for_1_or_2_year_service_player_3rd_year_maximum_if_first_2_year_maximum: bool = Field(
        description="If an Offer Sheet provides for the maximum allowable amount of Salary for the first two (2) Salary Cap Years, then the Offer Sheet may provide for Salary for the third Salary Cap Year of up to the maximum amount that the player would have been eligible to receive for the third Salary Cap Year."
    )
    offer_sheet_for_1_or_2_year_service_player_4th_year_maximum_if_3_year: bool = Field(
        description="If the Offer Sheet provides for Salary for the third Salary Cap Year, then the player’s Salary for the fourth Salary Cap Year may increase or decrease in relation to the third Salary Cap Year’s Salary by no more than 4.5% of the Salary for the third Salary Cap Year."
    )
    offer_sheet_for_1_or_2_year_service_player_average_salary_more_than_2_year: bool = Field(
        description="If a Team extends an Offer Sheet in accordance with Section 5(d)(ii) above, then, for purposes of determining whether the Team has Room for the Offer Sheet, the Salary for the first Salary Cap Year shall be deemed to equal the average of the aggregate Salaries for all Salary Cap Years covered by the Offer Sheet."
    )

    # first-round draft pick trade rules
    stepien_rule_no_sell_or_no_consecutive_first_round_draft_pick_trade: bool = Field(
        description="A Team cannot sell its first-round draft picks of any NBA Draft for cash or its equivalent, or trade or exchange its first-round draft picks of any NBA Draft if the result of such trade or exchange may be to leave the Member without first-round picks in any two (2) consecutive future NBA Drafts."
    )


class Response(BaseModel):
    answer: bool
    applied_rules: RuleExtraction = Field(
        description="If one of the rules are involved in the provided analysis, you should output True for it, otherwise False."
    )


prompt_template = """
Assume:
* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;
* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;
* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;
* the Luxury Tax is $170,814,000;
* the First Apron Level is $178,132,000;
* the Second Apron Level is $188,931,000;
* the Team Salary of each team listed under "Team Situations:" do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.

Reference Rules in NBA Collective Bargaining Agreement:

$reference_rules

Decide whether operations of any team violate the rules:

$question

Analyze the described scenario step-by-step and explicitly state the type of exception if you need to use any. End your response with:
1. "Answer: False." if no operation violates the rules; or
2. "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules.
Analysis:
$response
Do not parse any rules in Team Situations and Player Situations. Only parse the response after "Analysis:".
"""

with open("reference_rules.txt", "r") as f:
    rules = "".join(f.readlines())


def parse_rule_application(query_prompt: str, response: str):
    prompt = prompt_template.replace("$reference_rules", rules)
    prompt = prompt.replace("$question", query_prompt)
    prompt = prompt.replace("$response", response)
    parsing_response = gpt(prompt)
    applied_rules = [
        r for r, used in parsing_response.applied_rules.model_dump().items() if used
    ]
    return applied_rules


def build_query_prompt(query_dict: dict):
    team_info = "Team Situations:\n" + "\n".join(query_dict["team_situations"])
    player_info = "Player Situations:\n" + "\n".join(query_dict["player_situations"])
    operations = "Operations:\n" + "\n".join(query_dict["operations"])
    query_prompt = team_info + "\n\n" + player_info + "\n\n" + operations
    return query_prompt
