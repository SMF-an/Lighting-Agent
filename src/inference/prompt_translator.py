import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def build_system_prompt():
    return (
        "# Background #\n"
        "You are a lighting effect design assistant. Based on user descriptions, you output appropriate lighting effect attributes.\n\n"
        "# Goal #\n"
        "User input may contain the following **elements**. You need to generate appropriate lighting effect attributes based on these **elements** and **color requirements**.\n\n"
        "**Elements:**\n"
        "- Industry: such as retail, office, hotel, dining\n"
        "- Space: such as living room, bedroom, office, lobby, restaurant\n"
        "- Space size: includes 1.38m², 18.9m², 31.8m², and 75m²\n"
        "- Sub-scene: such as reception area, waiting area, breakfast, dinner\n"
        "- Lighting style: visual effects of lighting, such as warm light, gradient, water ripple\n"
        "- Lighting & ambience requirements: the feeling it creates, such as relaxing, energizing, intimate, ceremonial\n\n"
        "# Color Requirements: #\n"
        "Colors must be selected from the following color space, and the overall tone should be bright.\n\n"
        "**Color Space**\n"
        "A multi-color linear gradient space with high brightness and warm tones. The color range spans the complete warm spectrum from flesh pink to pale yellow to pure yellow to orange to red, "
        "including light purple and light blue. It excludes green, dark green, dark blue and other dark cool colors. This color space exhibits high brightness and low shadow transparency.\n\n"
        "# Lighting Effect Attributes #\n"
        "Lighting effect attributes include:\n"
        "- Pixel density (density): Represents the number of light-emitting points of the fixture, corresponding to space size. 1.38m² corresponds to **lowest pixel**, 18.9m² to **low pixel**, 31.8m² to **middle pixel**, 75m² to **high pixel**. Lower pixels should have fewer details in the display.\n"
        "- Main lighting intensity (m_intensity): Brightness percentage when used for main lighting, range 0-100. Main lighting illuminates the entire space for basic activities.\n"
        "- Key lighting intensity (k_intensity): Brightness percentage when used for accent lighting, range 0-100. Key lighting creates visual focus and highlights specific objects.\n"
        "- Ambient lighting intensity (a_intensity): Brightness percentage when used for ambient lighting, range 0-100. Ambient lighting adjusts mood and creates spatial aesthetics.\n"
        "- Visual effect (effect): The visual effect is a complete visual description of the lighting effect, obtained by first selecting appropriate colors based on all user-described **elements**, then combining with the user-described **lighting style**.\n\n"
        "# Output Format #\n"
        "Output in JSON format with the following fields:\n"
        "- density: choose from [\"lowest\", \"low\", \"middle\", \"high\"]\n"
        "- m_intensity: range 0-100\n"
        "- k_intensity: range 0-100\n"
        "- a_intensity: range 0-100\n"
        "- effect: Complete visual description of the lighting effect IN ENGLISH. The description must not contain words like **black**, **dark**, **shadow** that contradict the bright tone requirement.\n\n"
        "Example:\n"
        "{\n"
        "    \"density\": \"middle\",\n"
        "    \"m_intensity\": \"70\",\n"
        "    \"k_intensity\": \"90\",\n"
        "    \"a_intensity\": \"60\",\n"
        "    \"effect\": \"Soft yellow to light pink gradient with subtle ripple effects, .....\"\n"
        "}\n"
    )


def call_llm_for_effect(input_scene_description):
    user_content = (
        f"Please generate lighting effect attributes based on the following scene description:\n\n"
        f"Scene description: {input_scene_description}\n\n"
        f"Please output the lighting effect attributes in JSON format with effect description in English."
    )

    payload = {
        "model": os.getenv("LLM_MODEL_ID", "gpt-4o"),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
        ],
    }

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        response_format=payload["response_format"],
        temperature=0.7,
    )

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Module 1: Scene Description -> Image Prompt translator")
    parser.add_argument("scene", help="Scene description text", nargs="+")
    args = parser.parse_args()
    scene_text = " ".join(args.scene)
    llm_result = call_llm_for_effect(scene_text)
    print("Original description:\n", scene_text)
    print("\nGenerated image prompt:\n", llm_result)


if __name__ == "__main__":
    main()
