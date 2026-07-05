import pytest
import yaml

from evalloop import paths as paths_mod
from evalloop.schemas import load_task


def test_init_task_workspace_scaffolds_loadable_task(isolated_root):
    tp = paths_mod.init_task_workspace("my-task", root=isolated_root, answer_type="label")

    assert tp.task_config.exists()
    assert tp.prompt_file.exists()
    assert "{{input}}" in tp.prompt_file.read_text(encoding="utf-8")
    assert (tp.task_dir / "PROVENANCE.md").exists()
    assert not tp.golden.exists()  # the dataset is the user's job (and gitignored)
    assert not tp.rubric_file.exists()  # label task: no rubric scaffold
    assert "my-task" in paths_mod.list_tasks(isolated_root)

    # with a global config in place, the scaffold loads as-is (placeholder
    # labels keep TaskConfig's label-required validation satisfied)
    (isolated_root / "config.yaml").write_text(
        yaml.safe_dump({"default_task": "my-task", "models": [{"provider": "ollama:chat:q", "alias": "qwen7b"}]}),
        encoding="utf-8",
    )
    cfg, loaded = load_task("my-task", root=isolated_root)
    assert cfg.task.answer_type == "label"
    assert cfg.task.labels
    assert loaded.task_dir == tp.task_dir


def test_init_text_task_scaffolds_rubric(isolated_root):
    tp = paths_mod.init_task_workspace("t-text", root=isolated_root, answer_type="text")
    assert tp.rubric_file.exists()
    rubric = tp.rubric_file.read_text(encoding="utf-8")
    # the promptfoo placeholders must survive templating verbatim
    assert "{{input}}" in rubric and "{{expected}}" in rubric


def test_init_existing_task_raises(isolated_root):
    paths_mod.init_task_workspace("dup", root=isolated_root)
    with pytest.raises(paths_mod.TaskExistsError):
        paths_mod.init_task_workspace("dup", root=isolated_root)


def test_init_invalid_name_raises(isolated_root):
    with pytest.raises(paths_mod.TaskNotFoundError):
        paths_mod.init_task_workspace("Bad Name!", root=isolated_root)


def test_init_unknown_answer_type_raises(isolated_root):
    with pytest.raises(ValueError):
        paths_mod.init_task_workspace("t-x", root=isolated_root, answer_type="regex")
