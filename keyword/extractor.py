import os
import yake
import graphviz
import sys


def extract_keywords_yake(transcript, num_keywords=10):
    """Extract keywords using YAKE (single-document keyword extraction)."""
    print(f"\n🔹 Extracting keywords from transcript (length: {len(transcript)} characters)...")

    # Set YAKE parameters
    custom_kw_extractor = yake.KeywordExtractor(
        lan="en",   # Language: English
        n=2,        # Extract up to 2-word phrases (bigram keywords)
        dedupLim=0.9,  # Avoid duplicate similar keywords
        top=num_keywords
    )

    # Extract keywords
    keywords = custom_kw_extractor.extract_keywords(transcript)

    # Display results
    if not keywords:
        print("⚠️ Warning: No keywords extracted! Check input text.")
    else:
        print("\n✅ Extracted Keywords:")
        for word, score in keywords:
            print(f"  - {word}: {score:.4f}")

    return keywords


def create_keyword_flowchart_graphviz(keywords, output_file="keyword_flowchart.gv", view=True):
    """Create a flowchart from keywords based on their importance using Graphviz."""
    print(f"\n🛠️ Creating keyword flowchart with {len(keywords)} keywords using Graphviz...")

    if not keywords:
        print("⚠️ Warning: No keywords provided. Flowchart will be empty.")
        return None

    dot = graphviz.Digraph(comment='Keyword Flowchart')
    dot.attr(rankdir='TB', size='8,8')
    dot.attr('node', shape='box', style='filled', fontname='Arial')

    # Determine max score for coloring
    max_score = max([score for _, score in keywords]) if keywords else 1.0

    for keyword, score in keywords:
        color_intensity = int(255 * (score / max_score))
        color = f"#{255 - color_intensity:02x}{255 - color_intensity:02x}ff"
        dot.node(keyword, label=f"{keyword}\n({score:.4f})", fillcolor=color,
                 fontcolor='white' if color_intensity > 128 else 'black')

    # Connect keywords in order of importance
    sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_keywords) - 1):
        dot.edge(sorted_keywords[i][0], sorted_keywords[i + 1][0])

    try:
        dot.render(output_file, format='png', cleanup=False, view=view)
        print(f"✅ Flowchart saved to {output_file}.png")
    except Exception as e:
        print(f"❌ Error creating flowchart: {e}")

    return dot


def create_logical_flow_graphviz(transcript, keywords, output_file="logical_flow.gv", view=True):
    """Create a logical keyword flow based on keyword appearance in the text."""
    print(f"\n🛠️ Creating logical flow chart with {len(keywords)} keywords using Graphviz...")

    keyword_positions = {keyword: transcript.lower().find(keyword.lower()) for keyword, _ in keywords}
    keyword_positions = {k: v for k, v in keyword_positions.items() if v != -1}
    sorted_keywords = sorted(keyword_positions.items(), key=lambda x: x[1])

    dot = graphviz.Digraph(comment='Logical Keyword Flow')
    dot.attr(rankdir='TB', size='8,8')
    dot.attr('node', shape='box', style='filled', fontname='Arial')

    max_score = max([score for _, score in keywords]) if keywords else 1.0

    for keyword, position in sorted_keywords:
        score = next((s for k, s in keywords if k == keyword), 0.1)
        color_intensity = int(255 * (score / max_score))
        color = f"#99{255 - color_intensity:02x}{255 - color_intensity:02x}"
        dot.node(keyword, label=f"{keyword}\n({score:.4f})", fillcolor=color, fontcolor='black')

    for i in range(len(sorted_keywords) - 1):
        dot.edge(sorted_keywords[i][0], sorted_keywords[i + 1][0])

    try:
        dot.render(output_file, format='png', cleanup=True, view=view)
        print(f"✅ Logical flow chart saved to {output_file}.png")
    except Exception as e:
        print(f"❌ Error creating logical flow chart: {e}")

    return dot


def check_graphviz_installation():
    """Check if Graphviz is installed properly."""
    try:
        dot = graphviz.Digraph(comment='Test Graph')
        dot.node('A', 'Test Node')
        dot.render('test_graphviz', format='png', cleanup=True, view=False)
        print("✅ Graphviz test successful!")
        return True
    except Exception as e:
        print(f"❌ Graphviz test failed: {e}")
        print("🔹 Install Graphviz: https://graphviz.org/download/")
        return False


def check_environment():
    """Check the environment setup."""
    print("\n🔍 Environment check:")
    print(f"🟢 Python version: {sys.version}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"🛠️ Graphviz version: {graphviz.__version__}")
    print(f"📌 YAKE version: {yake.__version__}")


def main():
    """Main function to process transcript and generate flowcharts."""
    print("🚀 Starting keyword extraction and flowchart generation...")

    check_environment()

    if not check_graphviz_installation():
        print("❌ Please install Graphviz and try again.")
        return

    use_sample = input("🔹 Use sample transcript? (y/n): ").strip().lower() == 'y'

    if use_sample:
        transcript = """
        In today's meeting, we discussed the implementation of a new machine learning algorithm
        to improve our customer service. The data scientists presented their findings from the
        preliminary analysis of customer feedback. We need to collect more data to train our model
        effectively. The team agreed to use natural language processing techniques to analyze
        customer interactions. We will focus on sentiment analysis to gauge customer satisfaction.
        The project timeline was set for six months with monthly progress reports. The budget
        allocation was approved by the finance department. We also discussed the potential integration
        with our existing CRM system. The IT team will provide necessary infrastructure support.
        """
    else:
        transcript_file = input("📂 Enter transcript file path (or press Enter to type it manually): ").strip()
        if transcript_file and os.path.exists(transcript_file):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
        else:
            print("✍️ Type or paste your transcript text (Ctrl+D or Ctrl+Z+Enter to finish):")
            transcript_lines = []
            try:
                while True:
                    line = input()
                    transcript_lines.append(line)
            except EOFError:
                pass
            transcript = "\n".join(transcript_lines)

    print(f"\n📜 Transcript length: {len(transcript)} characters")

    num_keywords = int(input("🔹 Enter number of keywords to extract (default: 10): ").strip() or "10")

    # Extract keywords using YAKE
    tfidf_keywords = extract_keywords_yake(transcript, num_keywords=num_keywords)

    output_dir = input("📂 Enter output directory (default: current directory): ").strip() or "."
    os.makedirs(output_dir, exist_ok=True)

    view_files = input("🖼️ Open files when generated? (y/n): ").strip().lower() == 'y'

    keyword_chart_path = os.path.join(output_dir, "keyword_flowchart.gv")
    create_keyword_flowchart_graphviz(tfidf_keywords, output_file=keyword_chart_path, view=view_files)

    logical_flow_path = os.path.join(output_dir, "logical_flow.gv")
    create_logical_flow_graphviz(transcript, tfidf_keywords, output_file=logical_flow_path, view=view_files)

    print("\n✅ Flowcharts saved successfully!")


if __name__ == "__main__":
    main()
