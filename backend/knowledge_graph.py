
# knowledge_graph.py
# A simple local Knowledge Graph for CropSense AI.
# Relationships are educational, not confirmed diagnoses.

from crop_registry import UNSUPPORTED_CROPS, find_crops, matches_crop, normalize_text

#: The graph currently contains edges for these crops only. New supported crops
#: are recognised by the API guard but return no graph edges until data is added.
GRAPH_CROPS: tuple[str, ...] = ("tomato", "potato")

KNOWLEDGE_GRAPH = [
    {
        "subject": "Tomato",
        "relation": "may_show_symptom",
        "object": "Yellow leaves",
    },
    {
        "subject": "Yellow leaves",
        "relation": "may_indicate",
        "object": "Nutrient deficiency",
    },
    {
        "subject": "Yellow leaves",
        "relation": "may_be_associated_with",
        "object": "Overwatering",
    },
    {
        "subject": "Nutrient deficiency",
        "relation": "suggests_checking",
        "object": "Soil nutrients and growing conditions",
    },
    {
        "subject": "Potato",
        "relation": "may_show_symptom",
        "object": "Yellow leaves",
    },
    {
        "subject": "Potato",
        "relation": "may_show_symptom",
        "object": "Dark spots on leaves",
    },
    {
        "subject": "Dark spots on leaves",
        "relation": "requires_checking",
        "object": "Spot pattern and growing conditions",
    },
]


def search_knowledge_graph(question: str) -> list[dict]:
    """Return graph relationships relevant to the crop and symptom."""

    text = normalize_text(question)

    # Do not return graph relationships for crops the guide does not cover.
    if find_crops(text, UNSUPPORTED_CROPS):
        return []

    # Identify the requested crop. Crops without graph edges return [] rather
    # than inheriting another crop's relationships.
    crop = next(
        (candidate for candidate in GRAPH_CROPS if matches_crop(text, candidate)),
        None,
    )
    if crop is None:
        return []

    # Identify the requested symptom
    asks_yellow = "yellow" in text
    asks_spots = any(
        word in text
        for word in ["spot", "spots", "lesion", "lesions", "dark", "brown"]
    )

    results = []

    # Select relationships relevant to the requested crop and symptom
    for edge in KNOWLEDGE_GRAPH:
        subject = edge["subject"].lower()
        obj = edge["object"].lower()

        # Skip relationships explicitly about the other crop
        if crop == "tomato" and subject == "potato":
            continue
        if crop == "potato" and subject == "tomato":
            continue

        # Match crop-to-symptom relationships
        if crop and subject == crop:
            if asks_yellow and obj == "yellow leaves":
                results.append(edge)
                continue

            if asks_spots and "dark spots" in obj:
                results.append(edge)
                continue

        # Add useful cause/checking relationships for yellow leaves
        if asks_yellow and not asks_spots:
            if subject == "yellow leaves":
                results.append(edge)
                continue

            if subject == "nutrient deficiency":
                results.append(edge)
                continue

        # Add the checking relationship for dark spots
        if asks_spots and "dark spots" in subject:
            results.append(edge)

    # Remove duplicates while preserving order
    unique_results = []
    for edge in results:
        if edge not in unique_results:
            unique_results.append(edge)

    return unique_results


if __name__ == "__main__":
    questions = [
        "Why are my tomato leaves turning yellow?",
        "Why are my potato leaves turning yellow?",
        "Why does my potato plant have dark spots?",
    ]

    for question in questions:
        print("\nQuestion:", question)
        results = search_knowledge_graph(question)

        for edge in results:
            print(
                f'{edge["subject"]} '
                f'--[{edge["relation"]}]--> '
                f'{edge["object"]}'
            )
