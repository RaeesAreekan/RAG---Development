from rag_pipeline.config import Settings


def test_settings_have_expected_defaults() -> None:
    settings = Settings(_env_file=None) # type: ignore

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.generation_model == "gpt-4o"
    assert settings.evaluation_model == "gpt-4o-mini"
    assert settings.dense_top_k == 10
    assert settings.sparse_top_k == 10
    assert settings.final_top_k == 5


def test_settings_keep_dense_and_sparse_weights_configurable() -> None:
    settings = Settings(
        _env_file=None, # type: ignore
        dense_weight=0.8,
        sparse_weight=0.2,
    )

    assert settings.dense_weight == 0.8
    assert settings.sparse_weight == 0.2