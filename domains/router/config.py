"""
Configuration for the semantic router model.
"""

# Embedding model for semantic routing
# Good multilingual support for Swedish
ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Alternative options:
# - "intfloat/multilingual-e5-small" - Strong retrieval performance
# - "KBLab/sentence-bert-swedish-cased" - Swedish-specific (KB Lab)

# Route names
ROUTE_RIKSBANKEN = "riksbanken"
ROUTE_GENERAL = "general"

# Similarity threshold (queries below this go to general)
SIMILARITY_THRESHOLD = 0.5

# Data paths
RIKSBANKEN_EXAMPLES_PATH = "data/router/riksbanken_examples.jsonl"
GENERAL_EXAMPLES_PATH = "data/router/general_examples.jsonl"
TEST_EXAMPLES_PATH = "data/router/test_examples.jsonl"

# Artifacts (JSON format - router is reconstructed at load time)
ROUTER_ARTIFACT_PATH = "outputs/router/semantic_router.json"
