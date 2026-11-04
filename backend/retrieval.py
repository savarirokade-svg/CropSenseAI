
from pathlib import Path
import re

DATA_FOLDER = Path(__file__).parent / "data"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "from", "with", "and",
    "or", "but", "my", "me", "i", "it", "this", "that", "what",
    "why", "how", "can", "could", "should", "would", "do", "does",
    "if", "there", "their", "they", "them", "about", "please",
    "explain", "tell", "show", "give", "possible", "cause", "causes",
    "check", "turning", "turn", "become", "around", "appear",
    "have", "has", "plant", "leaves", "leaf"
}


def get_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return words - STOP_WORDS


def search_documents(question: str, top_k: int = 3):
    question_lower = question.lower()
    question_words = get_words(question)

    if not question_words:
        return []

    # Identify the crop mentioned in the question.
    crop = None

    if re.search(r"\btomato(?:es)?\b", question_lower):
        crop = "tomato"
    elif re.search(r"\bpotato(?:es)?\b", question_lower):
        crop = "potato"
        
    # Crops not supported by the current guide
    unsupported_crops = [
        "mango", "rice", "wheat", "banana",
        "cotton", "chilli", "chili", "grapes",
        "apple", "onion", "maize", "corn"
    ]

    for unsupported_crop in unsupported_crops:
        if re.search(
            rf"\b{re.escape(unsupported_crop)}\b",
            question_lower
        ):
            return []


    # Identify the symptom mentioned in the question.
    asks_yellow = bool(
        re.search(r"\byellow(?:ing)?\b", question_lower)
    )

    asks_spots = bool(
        re.search(
            r"\b(?:spots?|lesions?|brown|dark|blight)\b",
            question_lower
        )
    )

    results = []

    if not DATA_FOLDER.exists():
        return []

    for file_path in DATA_FOLDER.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        # Keep each SECTION heading with its description.
        sections = re.split(
            r"(?=SECTION\s+\d+\s*:)",
            text,
            flags=re.IGNORECASE
        )

        for section in sections:
            section = section.strip()

            if not section:
                continue

            if section.lower().startswith("document:"):
                continue

            if section.lower().startswith("important note"):
                continue

            lines = section.splitlines()
            heading = lines[0].strip().lower()
            section_lower = section.lower()

            # Identify the crop named in the section heading.
            section_crop = None

            if re.search(r"\btomato(?:es)?\b", heading):
                section_crop = "tomato"
            elif re.search(r"\bpotato(?:es)?\b", heading):
                section_crop = "potato"

            # Never return a section about the wrong crop.
            if crop is not None and section_crop is not None:
                if section_crop != crop:
                    continue

            is_general = "information to collect" in heading

            # Exclude unrelated symptom sections.
            if asks_yellow and not asks_spots:
                if re.search(
                    r"\b(?:spots?|lesions?|blight)\b",
                    heading
                ):
                    continue

            if asks_spots and not asks_yellow:
                if re.search(r"\byellow(?:ing)?\b", heading):
                    continue

            # Calculate the relevance score.
            section_words = get_words(section)
            matches = question_words.intersection(section_words)

            if not matches and not is_general:
                continue

            score = len(matches) / max(len(question_words), 1)

            # Give extra weight to the requested crop.
            if crop and section_crop == crop:
                score += 0.50

            # Give extra weight to matching symptoms.
            if asks_yellow and re.search(
                r"\byellow(?:ing)?\b", heading
            ):
                score += 0.50

            if asks_spots and re.search(
                r"\b(?:spots?|lesions?|brown|dark|blight)\b",
                heading
            ):
                score += 0.50

            # General guidance is useful but should rank lower.
            if is_general:
                score = min(score, 0.30)

            score = round(min(score, 1.0), 3)

            results.append({
                "source": file_path.name,
                "text": section,
                "score": score,
            })

    # Sort by highest relevance first.
   
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Return only results with meaningful relevance.
    relevant_results = [
        item for item in results
        if item["score"] >= 0.35
    ]

    # If nothing relevant matches, ask for more information.
    return relevant_results[:max(1, top_k)]



if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    results = search_documents(question)

    print("\nRelevant information:\n")

    if not results:
        print("No relevant information found.")
    else:
        for index, item in enumerate(results, start=1):
            print(
                f"Result {index} | Source: {item['source']}"
            )
            print(f"Match score: {item['score']}")
            print(item["text"])
            print("-" * 50)
