import numpy as np
import random

us_city_collection = [
    "Seattle",
    "Portland",
    "Los Angeles",
    "San Francisco",
    "Sacramento",
    "Salt Lake City",
    "Denver",
    "Las Vegas",
    "Phoenix",
    "Dallas",
    "Houston",
    "Austin",
    "Minneapolis",
    "New Orleans",
    "Chicago",
    "Charlotte",
    "Washington D.C.",
    "Philadelphia",
    "New York",
    "Boston",
    "Atlanta",
    "Orlando",
    "Miami",
]

international_city_collection = {
    "Shanghai": "China",
    "Beijing": "China",
    "Wuhan": "China",
    "Guangzhou": "China",
    "Shenzhen": "China",
    "Chengdu": "China",
    "Xiamen": "China",
    "Hong Kong": "Hong Kong",
    "Tokyo": "Japan",
    "Osaka": "Japan",
    "Kyoto": "Japan",
    "Nagoya": "Japan",
    "Seoul": "South Korea",
    "Busan": "South Korea",
    "Incheon": "South Korea",
    "New Delhi": "India",
    "Mumbai": "India",
    "Doha": "Qatar",
    "Sydney": "Australia",
    "Melbourne": "Australia",
    "Auckland": "New Zealand",
    "London": "Europe",
    "Paris": "Europe",
    "Madrid": "Europe",
    "Barcelona": "Europe",
    "Roma": "Europe",
    "Amsterdam": "Europe",
    "Stockholm": "Europe",
    "Brussels": "Europe",
    "Zurich": "Europe",
    "Helsinki": "Europe",
    "Berlin": "Europe",
    "Munich": "Europe",
    "Athens": "Europe",
    "Vienna": "Europe",
    "Copenhagen": "Europe",
    "Port-au-Prince": "Haiti",
    "Mexico City": "Mexico",
    "Toronto": "Canada",
    "Montreal": "Canada",
    "Vancouver": "Canada",
    "Ottawa": "Canada",
    "Havana": "Cuba",
    "Panama City": "Panama",
    "San Juan": "Puerto Rico",
    "Bogotá": "Colombia",
    "Quito": "Ecuador",
    "Lima": "Peru",
    "Buenos Aires": "South America",
    "Rio de Janeiro": "South America",
    "São Paulo": "South America",
    "Jerusalem": "Israel",
}

base_price_interval = {
    "U.S.": (30, 300),
    "China": (500, 1500),
    "Hong Kong": (800, 1500),
    "Japan": (300, 600),
    "South Korea": (300, 600),
    "India": (300, 800),
    "Qatar": (700, 1200),
    "Europe": (150, 700),
    "Canada": (100, 400),
    "Haiti": (50, 100),
    "Mexico": (80, 200),
    "Cuba": (150, 200),
    "Panama": (150, 200),
    "Puerto Rico": (150, 300),
    "Colombia": (150, 300),
    "Ecuador": (300, 600),
    "Peru": (250, 700),
    "South America": (400, 1200),
    "Israel": (700, 1500),
    "Australia": (450, 1000),
    "New Zealand": (800, 1000),
}

bag_types = ["backpack", "luggage box"]
customer_types = [
    "Basic Economy",
    "Main Cabin",
    "Main Plus",
    "Premium Economy",
    "Business",
    "First",
]
name_collection = [
    "James",
    "John",
    "William",
    "Robert",
    "Michael",
    "David",
    "Joseph",
    "Charles",
    "Thomas",
    "Daniel",
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
    "Emily",
]

prompt_template = """$name is a $class Class passenger flying from $departure to $destination with the following items:
$bag_list

$name's flight ticket is $base_price."""


def sample_carry_on():
    length = np.random.randint(18, 23)
    width = np.random.randint(11, 15)
    height = np.random.randint(6, 10)
    weight = np.random.randint(5, 13)
    return length, width, height, weight


def gen_question(complexity: int):
    assert complexity in [0, 1, 2]
    num_bags = [5, 8, 11][complexity]
    customer_name = random.choice(name_collection)
    customer_class = random.choice(customer_types)
    direction = np.random.randint(2)
    all_cities = us_city_collection + list(international_city_collection.keys())
    if direction == 0:
        departure = random.choice(us_city_collection)
        all_cities.remove(departure)
        destination = random.choice(all_cities)
        region = international_city_collection.get(destination, "U.S.")
    else:
        destination = random.choice(us_city_collection)
        all_cities.remove(destination)
        departure = random.choice(all_cities)
        region = international_city_collection.get(departure, "U.S.")
    # NOTE: Different base price for different regions
    base_price = np.random.randint(*base_price_interval[region])
    prompt = prompt_template.replace("$name", customer_name)
    prompt = prompt.replace("$class", customer_class)
    prompt = prompt.replace("$departure", departure)
    prompt = prompt.replace("$destination", destination)
    prompt = prompt.replace("$base_price", "$" + str(base_price))

    bag_list, prompt_bag_list = [], []
    # Generate one carry-on item
    length, width, height, weight = sample_carry_on()
    bag_list.append(
        {
            "id": 1,
            "name": "backpack",
            "size": [length, width, height],
            "weight": weight,
        }
    )
    prompt_bag_list.append(
        f"1. A backpack: {length} x {width} x {height} inches, {weight} lbs;"
    )

    # NOTE: Fix bag limitation from/to Cuba
    if region == "Cuba":
        num_bags = min(3, num_bags)
    elif region in ["South America", "Brazil", "Mexico"]:
        num_bags = min(6, num_bags)

    # Randomly sample bags
    for bag_id in range(1, num_bags):
        bag_class = random.choice(bag_types)
        overweight = oversize = True

        if overweight:
            if region in [
                "Cuba",
                "Europe",
                "Israel",
                "Qatar",
                "Australia",
                "New Zealand",
            ]:
                interval = random.choice([(51, 54), (54, 71)])
            else:
                interval = random.choice([(51, 54), (54, 71), (71, 101)])
            weight = np.random.randint(*interval)
        else:
            weight = np.random.randint(30, 51)

        if oversize:
            interval = random.choice([(63, 66), (66, 116)])
            size = np.random.randint(*interval)
        else:
            size = np.random.randint(45, 63)

        height = np.random.randint(size // 3 - 12, size // 3 - 6)
        length = np.random.randint(size // 3 + 12, size // 3 + 18)
        width = size - height - length

        bag_list.append(
            {
                "id": bag_id + 1,
                "name": bag_class,
                "size": [length, width, height],
                "weight": weight,
            }
        )
        prompt_bag_list.append(
            f"{bag_id + 1}. A {bag_class}: {length} x {width} x {height} inches, {weight} lbs;"
        )

    prompt_bag_list = "\n".join(prompt_bag_list)
    prompt = prompt.replace("$bag_list", prompt_bag_list)

    info_dict = {
        "base_price": base_price,
        "customer_class": customer_class,
        "routine": region,
        "direction": direction,
        "bag_list": bag_list,
    }

    return prompt, info_dict


if __name__ == "__main__":
    from pprint import pprint

    prompt, info_dict = gen_question(3, 3)
    print(prompt)
    pprint(info_dict)
