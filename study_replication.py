#!/usr/bin/env python3
"""
Art Study Replication: LLM-Based Aesthetic Evaluation

Replicates aspects of the Mandel et al. "Beauty or Money" study using 
3 different LLMs (or LLM personas) to evaluate paintings on aesthetic criteria.

Usage:
    python study_replication.py [--mock]

Options:
    --mock    Use simulated LLM responses for testing (no API calls)
"""

import os
import sys
import json
import base64
import random
import pandas as pd
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Try to import API clients (optional, will use mock if not available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Configuration
SCRIPT_DIR = Path(__file__).parent
PAINTINGS_DIR = SCRIPT_DIR / "paintings-10"
METADATA_FILE = SCRIPT_DIR / "1000_paintings features.xlsx"
NUM_PAINTINGS = 10
RANDOM_SEED = 42

# Cost tracking (per 1M tokens, in USD)
COST_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}
TOTAL_COST = {"input_tokens": 0, "output_tokens": 0, "usd": 0.0}

# Evaluation criteria (from Mandel et al. paper)
CRITERIA = [
    "aesthetic",    # General impression
    "harmony",      # Visual harmony
    "relaxation",   # Feeling of relaxation
    "technique",    # Technical skill
    "hedonic",      # Pleasure/enjoyment
    "arousal"       # Emotional arousal/excitement
]

EVALUATION_PROMPT = """You are an art evaluator. Please rate this painting on a scale of 1-10 for each criterion.

**Criteria:**
1. **Aesthetic** (1-10): Your general aesthetic impression of this artwork.
2. **Harmony** (1-10): The visual harmony, balance, and coherence of the composition.
3. **Relaxation** (1-10): How relaxing or calming this artwork feels to view.
4. **Technique** (1-10): The technical skill and craftsmanship displayed.
5. **Hedonic** (1-10): The pleasure and enjoyment you derive from viewing this artwork.
6. **Arousal** (1-10): How emotionally arousing, exciting, or stimulating this artwork is.

**Important:** 
- Rate based ONLY on what you see in the image, without any assumptions about the artist, period, or value.
- Provide your response as a JSON object with exactly these keys: "aesthetic", "harmony", "relaxation", "technique", "hedonic", "arousal"
- Each value must be an integer from 1 to 10.

Example response format:
{"aesthetic": 7, "harmony": 6, "relaxation": 5, "technique": 8, "hedonic": 6, "arousal": 4}

Now, please evaluate the painting shown in the image."""


def load_image_as_base64(image_path: Path) -> str:
    """Load an image file and return it as base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_random_paintings(df: pd.DataFrame, n: int = NUM_PAINTINGS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Select n random paintings from the dataset."""
    random.seed(seed)
    # Use the 'As in Last_Art_Results' column as the image filename reference
    valid_df = df.dropna(subset=['As in Last_Art_Results'])
    
    # Verify that images exist
    valid_indices = []
    for idx, row in valid_df.iterrows():
        img_name = row['As in Last_Art_Results']
        img_path = PAINTINGS_DIR / img_name
        if img_path.exists():
            valid_indices.append(idx)
    
    # Random sample
    sample_indices = random.sample(valid_indices, min(n, len(valid_indices)))
    return df.loc[sample_indices].copy()


def parse_llm_response(response_text: str) -> dict:
    """Parse LLM response to extract ratings."""
    # Try to find JSON in the response
    try:
        # Look for JSON object in the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            ratings = json.loads(json_str)
            # Validate and clamp values
            result = {}
            for criterion in CRITERIA:
                value = ratings.get(criterion, 5)
                result[criterion] = max(1, min(10, int(value)))
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: return neutral ratings
    return {c: 5 for c in CRITERIA}


# ============ LLM Evaluator Classes ============

class MockEvaluator:
    """Simulated LLM for testing without API calls."""
    
    def __init__(self, name: str, bias: dict = None):
        self.name = name
        self.bias = bias or {}
    
    def evaluate(self, image_path: Path) -> dict:
        """Return simulated ratings with some randomness."""
        random.seed(hash(str(image_path) + self.name))
        ratings = {}
        for criterion in CRITERIA:
            base = random.randint(4, 8)
            bias = self.bias.get(criterion, 0)
            ratings[criterion] = max(1, min(10, base + bias))
        return ratings


class OpenAIEvaluator:
    """OpenAI GPT-4o-mini evaluator (cost-effective)."""
    
    def __init__(self, name: str = "gpt-4o-mini"):
        self.name = name
        self.model_name = "gpt-4o-mini"
        self.client = openai.OpenAI()
    
    def evaluate(self, image_path: Path) -> dict:
        global TOTAL_COST
        image_data = load_image_as_base64(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EVALUATION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # Track costs
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * COST_PER_1M_TOKENS[self.model_name]["input"] + 
                    output_tokens * COST_PER_1M_TOKENS[self.model_name]["output"]) / 1_000_000
            TOTAL_COST["input_tokens"] += input_tokens
            TOTAL_COST["output_tokens"] += output_tokens
            TOTAL_COST["usd"] += cost
        
        return parse_llm_response(response.choices[0].message.content)


class AnthropicEvaluator:
    """Anthropic Claude 3 Haiku evaluator (cost-effective)."""
    
    def __init__(self, name: str = "claude-3-haiku"):
        self.name = name
        self.model_name = "claude-3-haiku-20240307"
        self.client = anthropic.Anthropic()
    
    def evaluate(self, image_path: Path) -> dict:
        global TOTAL_COST
        image_data = load_image_as_base64(image_path)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": EVALUATION_PROMPT}
                    ]
                }
            ]
        )
        
        # Track costs
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * COST_PER_1M_TOKENS["claude-3-haiku"]["input"] + 
                    output_tokens * COST_PER_1M_TOKENS["claude-3-haiku"]["output"]) / 1_000_000
            TOTAL_COST["input_tokens"] += input_tokens
            TOTAL_COST["output_tokens"] += output_tokens
            TOTAL_COST["usd"] += cost
        
        return parse_llm_response(response.content[0].text)


class GeminiEvaluator:
    """Google Gemini evaluator."""
    
    def __init__(self, name: str = "gemini"):
        self.name = name
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def evaluate(self, image_path: Path) -> dict:
        import PIL.Image
        import time
        
        image = PIL.Image.open(image_path)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content([EVALUATION_PROMPT, image])
                return parse_llm_response(response.text)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 60  # Wait 60 seconds on rate limit
                    print(f"\n      Rate limited, waiting {wait_time}s...", end="", flush=True)
                    time.sleep(wait_time)
                    print(" retrying...", end=" ", flush=True)
                else:
                    raise


def create_evaluators(use_mock: bool = False) -> list:
    """Create the 3 LLM evaluators based on availability."""
    if use_mock:
        print("Using MOCK evaluators (no API calls)")
        return [
            MockEvaluator("evaluator_A"),
            MockEvaluator("evaluator_B"),
            MockEvaluator("evaluator_C")
        ]
    
    evaluators = []
    
    # Try to add real evaluators based on availability
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        evaluators.append(OpenAIEvaluator("gpt-4o-mini"))
        print("✓ OpenAI GPT-4o-mini evaluator added")
    
    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        evaluators.append(AnthropicEvaluator("claude-3-haiku"))
        print("✓ Anthropic Claude-3-Haiku evaluator added")
    
    if GEMINI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
        evaluators.append(GeminiEvaluator("gemini-2.0-flash"))
        print("✓ Google Gemini-2.0-Flash evaluator added")
    
    # If we don't have 3 real evaluators, pad with mock ones
    if len(evaluators) < 3:
        print(f"\n⚠️  Only {len(evaluators)} API evaluator(s) available. Adding mock evaluators.")
        mock_count = 3 - len(evaluators)
        for i in range(mock_count):
            evaluators.append(MockEvaluator(f"mock_{i+1}"))
    
    return evaluators[:3]


def run_study(use_mock: bool = False):
    """Run the complete study replication."""
    print("=" * 60)
    print("Art Study Replication: LLM-Based Aesthetic Evaluation")
    print("=" * 60)
    print()
    
    # Load metadata
    print("Loading painting metadata...")
    df = pd.read_excel(METADATA_FILE)
    print(f"  Total paintings in dataset: {len(df)}")
    
    # Select random paintings
    print(f"\nSelecting {NUM_PAINTINGS} random paintings...")
    sample_df = get_random_paintings(df, NUM_PAINTINGS)
    print(f"  Selected paintings:")
    for _, row in sample_df.iterrows():
        print(f"    - {row['Title'][:50]}... by {row['Artist']}")
    
    # Create evaluators
    print("\nInitializing evaluators...")
    evaluators = create_evaluators(use_mock)
    print(f"  Using evaluators: {[e.name for e in evaluators]}")
    
    # Run evaluations
    print("\n" + "=" * 60)
    print("Running evaluations...")
    print("=" * 60)
    
    results = []
    
    for idx, (_, painting) in enumerate(sample_df.iterrows()):
        img_name = painting['As in Last_Art_Results']
        img_path = PAINTINGS_DIR / img_name
        
        print(f"\n[{idx+1}/{NUM_PAINTINGS}] {painting['Title'][:40]}...")
        
        for evaluator in evaluators:
            print(f"  → {evaluator.name}...", end=" ", flush=True)
            
            try:
                ratings = evaluator.evaluate(img_path)
                print(f"✓ ({', '.join(str(ratings[c]) for c in CRITERIA[:3])})")
            except Exception as e:
                print(f"✗ Error: {e}")
                ratings = {c: None for c in CRITERIA}
            
            # Parse price (remove commas and convert)
            try:
                price_str = str(painting['Price, $']).replace(',', '').replace('$', '')
                price = float(price_str) if price_str and price_str != 'nan' else None
            except:
                price = None
            
            results.append({
                'painting_id': painting['Codes'],
                'image_file': img_name,
                'title': painting['Title'],
                'artist': painting['Artist'],
                'year_made': painting.get('Year of making'),
                'price_usd': price,
                'evaluator': evaluator.name,
                **ratings
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = SCRIPT_DIR / "results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\n✓ Results saved to: {output_file}")
    
    # Generate analysis summary
    generate_analysis(results_df)
    
    # Print cost summary
    print(f"\n💰 Total API Cost:")
    print(f"   Input tokens:  {TOTAL_COST['input_tokens']:,}")
    print(f"   Output tokens: {TOTAL_COST['output_tokens']:,}")
    print(f"   Total cost:    ${TOTAL_COST['usd']:.4f}")
    
    return results_df, TOTAL_COST


def generate_analysis(results_df: pd.DataFrame):
    """Generate analysis summary markdown file."""
    print("\nGenerating analysis summary...")
    
    # Calculate summary statistics
    numeric_cols = CRITERIA
    
    summary_lines = [
        "# Art Study Replication: Analysis Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Paintings evaluated:** {results_df['painting_id'].nunique()}",
        f"- **Evaluators:** {', '.join(results_df['evaluator'].unique())}",
        f"- **Total evaluations:** {len(results_df)}",
        "",
        "## Criteria",
        "",
    ]
    
    # Add criteria descriptions
    criteria_desc = {
        "aesthetic": "General aesthetic impression",
        "harmony": "Visual harmony and balance",
        "relaxation": "Feeling of relaxation",
        "technique": "Technical skill displayed",
        "hedonic": "Pleasure and enjoyment",
        "arousal": "Emotional arousal/excitement"
    }
    for i, c in enumerate(CRITERIA, 1):
        desc = criteria_desc.get(c, c.title())
        summary_lines.append(f"{i}. **{c.title()}** - {desc}")
    
    summary_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "### By Evaluator",
        "",
        "| Evaluator | " + " | ".join([c.title() for c in CRITERIA]) + " |",
        "|-----------|" + "|".join(["----" for _ in CRITERIA]) + "|",
    ])
    
    for evaluator in results_df['evaluator'].unique():
        eval_df = results_df[results_df['evaluator'] == evaluator]
        row = f"| {evaluator} | "
        for col in numeric_cols:
            mean = eval_df[col].mean()
            std = eval_df[col].std()
            row += f"{mean:.1f} ± {std:.1f} | "
        summary_lines.append(row)
    
    summary_lines.extend([
        "",
        "### Overall Means",
        "",
    ])
    
    for col in numeric_cols:
        mean = results_df[col].mean()
        std = results_df[col].std()
        summary_lines.append(f"- **{col.replace('_', ' ').title()}:** {mean:.2f} ± {std:.2f}")
    
    # Correlation with price (if available)
    valid_price_df = results_df.dropna(subset=['price_usd'])
    if len(valid_price_df) > 5:
        summary_lines.extend([
            "",
            "## Correlation with Price",
            "",
            "| Criterion | Correlation (r) |",
            "|-----------|-----------------|",
        ])
        
        for col in numeric_cols:
            try:
                corr = valid_price_df[[col, 'price_usd']].corr().iloc[0, 1]
                summary_lines.append(f"| {col.replace('_', ' ').title()} | {corr:.3f} |")
            except:
                summary_lines.append(f"| {col.replace('_', ' ').title()} | N/A |")
    
    # Individual painting results
    summary_lines.extend([
        "",
        "## Individual Painting Results",
        "",
        "| Painting | Artist | Price ($) | " + " | ".join([c.title() for c in CRITERIA]) + " |",
        "|----------|--------|-----------|" + "|".join(["----" for _ in CRITERIA]) + "|",
    ])
    
    for painting_id in results_df['painting_id'].unique():
        p_df = results_df[results_df['painting_id'] == painting_id]
        title = p_df['title'].iloc[0][:25] + "..." if len(p_df['title'].iloc[0]) > 25 else p_df['title'].iloc[0]
        artist = p_df['artist'].iloc[0][:12] if pd.notna(p_df['artist'].iloc[0]) else "Unknown"
        price = p_df['price_usd'].iloc[0]
        price_str = f"{price:,.0f}" if pd.notna(price) else "N/A"
        
        avg_scores = [f"{p_df[c].mean():.1f}" for c in CRITERIA]
        
        summary_lines.append(
            f"| {title} | {artist} | {price_str} | " + " | ".join(avg_scores) + " |"
        )
    
    # Key findings
    summary_lines.extend([
        "",
        "## Key Findings",
        "",
        "_Note: This is a simplified replication using LLMs instead of human participants._",
        "_Results should be interpreted as exploratory rather than definitive._",
        "",
    ])
    
    # Write to file
    output_file = SCRIPT_DIR / "analysis_summary.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Analysis saved to: {output_file}")


if __name__ == "__main__":
    use_mock = "--mock" in sys.argv
    
    if not use_mock:
        # Check for API keys
        has_keys = any([
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("ANTHROPIC_API_KEY"),
            os.environ.get("GOOGLE_API_KEY")
        ])
        
        if not has_keys:
            print("⚠️  No API keys found in environment variables.")
            print("   Set one or more of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
            print("   Or run with --mock flag for simulated results.")
            print()
            response = input("Continue with mock evaluators? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                sys.exit(0)
            use_mock = True
    
    results = run_study(use_mock=use_mock)
    
    print("\n" + "=" * 60)
    print("Study complete!")
    print("=" * 60)
