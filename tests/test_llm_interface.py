from llmbo.llm_interface import LLMInterface


def test_build_candidate_prompt():
    iface = LLMInterface(api_key="test-key", base_url="http://test.local", model="test-model")
    history = [([-0.5, 1.2], 0.85), ([0.3, -0.8], 0.72)]
    param_bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    prompt = iface._build_candidate_prompt(history, param_bounds, n_candidates=2)
    assert "-0.5" in prompt
    assert "0.85" in prompt
    assert "Best so far" in prompt


def test_parse_candidates_valid_json():
    iface = LLMInterface(api_key="test-key", base_url="http://test.local", model="test-model")
    response = '{"candidates": [[0.11, -0.32], [0.87, 0.45]], "reasoning": "exploring"}'
    candidates = iface._parse_candidates(response, n_dims=2)
    assert len(candidates) == 2
    assert len(candidates[0]) == 2
    assert abs(candidates[0][0] - 0.11) < 1e-6


def test_parse_candidates_with_markdown():
    iface = LLMInterface(api_key="test-key", base_url="http://test.local", model="test-model")
    response = '```json\n{"candidates": [[0.5, 0.5]], "reasoning": "test"}\n```'
    candidates = iface._parse_candidates(response, n_dims=2)
    assert len(candidates) == 1
