from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

from app import studio_migrate
from app.pipeline import _build_input_hash
from app.studio_profiles import (
    ProfileError,
    ProfileStore,
    export_profile_bundle,
    import_profile_bundle,
    scan_workflow,
    snapshot_hash,
)
from app.studio_settings import (
    SecretStore,
    SettingsStore,
    StudioSettingsError,
    get_studio_paths,
)
from app.studio_providers import ProviderTestError, test_subtitle_profile


class StudioSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.appdata = self.root / "appdata"
        self.workspace = self.root / "workspace"
        self.env = mock.patch.dict(
            os.environ,
            {
                "AI_VIDEO_APPDATA": str(self.appdata),
                "AI_VIDEO_WORKSPACE": str(self.workspace),
            },
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_settings_initialize_creates_separate_workspace_tree(self) -> None:
        settings = SettingsStore().initialize(
            self.workspace, jianying_draft_root=self.root / "jianying"
        )
        self.assertTrue(settings["initialized"])
        paths = get_studio_paths()
        self.assertEqual(paths.workspace, self.workspace.resolve())
        for name in (
            "projects",
            "outputs",
            "raw",
            "cache",
            "runtime",
            "workflows",
            "profiles",
            "fonts",
            "exports",
        ):
            self.assertTrue((self.workspace / name).is_dir(), name)

    @unittest.skipUnless(os.name == "nt", "DPAPI is a Windows feature")
    def test_dpapi_secret_roundtrip_never_writes_plaintext(self) -> None:
        store = SecretStore()
        value = "test-secret-value-that-must-not-leak"
        store.set("DEEPSEEK_API_KEY", value)
        self.assertEqual(store.get("DEEPSEEK_API_KEY"), value)
        self.assertNotIn(value.encode("utf-8"), store.path.read_bytes())
        self.assertEqual(store.status()["DEEPSEEK_API_KEY"]["configured"], True)

    def test_relative_workspace_is_rejected(self) -> None:
        settings = SettingsStore().load()
        settings["workspace"] = "relative"
        with self.assertRaisesRegex(StudioSettingsError, "绝对路径"):
            SettingsStore().save(settings)


class StudioProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(
            os.environ,
            {
                "AI_VIDEO_APPDATA": str(self.root / "appdata"),
                "AI_VIDEO_WORKSPACE": str(self.root / "workspace"),
            },
        )
        self.env.start()
        SettingsStore().initialize(self.root / "workspace", jianying_draft_root=self.root / "draft")
        self.paths = get_studio_paths()
        self.store = ProfileStore(self.paths)

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_builtins_include_all_five_profile_kinds(self) -> None:
        self.assertIn("deepseek_default", self.store.list("llm"))
        self.assertIn("comfyui_default", self.store.list("image"))
        self.assertIn("history_image_default", self.store.list("comfyui_workflow"))
        self.assertIn("yunyang_soft", self.store.list("voice"))
        self.assertIn("social_pink", self.store.list("subtitle"))

    def test_snapshot_is_stable_and_has_no_secret_values(self) -> None:
        snapshot = self.store.snapshot(
            {
                "llm": "llm:deepseek_default",
                "image": "image:comfyui_default",
                "voice": "voice:yunyang_soft",
                "subtitle": "subtitle:social_pink",
            }
        )
        self.assertEqual(snapshot["sha256"], snapshot_hash(snapshot))
        self.assertNotIn("test-secret", json.dumps(snapshot))

    def test_external_image_requires_fixed_integer_limit(self) -> None:
        with self.assertRaisesRegex(ProfileError, "固定单次上限"):
            self.store.save(
                {
                    "kind": "image",
                    "id": "bad_image",
                    "name": "Bad",
                    "protocol": "images_compatible",
                    "base_url": "https://example.com/v1",
                    "model": "image",
                    "max_images_per_run": "all_candidates",
                }
            )

    def test_workflow_scanner_finds_standard_bindings_and_models(self) -> None:
        workflow = json.loads(
            (Path(__file__).resolve().parents[1] / "history_image_api.json").read_text(
                encoding="utf-8"
            )
        )
        scan = scan_workflow(workflow)
        self.assertEqual(scan["unresolved"], [])
        self.assertEqual(scan["bindings"]["prompt"], ["57:27", "text"])
        self.assertTrue(scan["models"]["unet"])
        self.assertTrue(scan["samplers"])

    def test_export_excludes_secret_refs_and_absolute_paths(self) -> None:
        profile = self.store.save(
            {
                "kind": "llm",
                "id": "my_deepseek",
                "name": "Mine",
                "protocol": "chat_completions",
                "base_url": "https://api.example.com",
                "model": "example",
                "timeout_seconds": 60,
                "max_tokens": 1000,
                "secret_ref": "DEEPSEEK_API_KEY",
                "local_path": str(self.root / "private"),
            }
        )
        bundle = export_profile_bundle(self.root / "profiles.zip", self.store)
        with zipfile.ZipFile(bundle) as archive:
            text = "\n".join(
                archive.read(name).decode("utf-8")
                for name in archive.namelist()
                if name.endswith(".json")
            )
        self.assertNotIn("secret_ref", text)
        self.assertNotIn(str(self.root), text)

    def test_imported_bundle_requires_revalidation(self) -> None:
        self.store.save(
            {
                "kind": "voice",
                "id": "custom_voice",
                "name": "Custom",
                "provider": "edge_tts",
                "voice": "zh-CN-YunyangNeural",
                "rate": "+1%",
                "pitch": "+0Hz",
                "validated": True,
            }
        )
        bundle = export_profile_bundle(self.root / "voices.zip", self.store)
        imported = import_profile_bundle(bundle, self.store)
        self.assertFalse(imported[0]["validated"])

    def test_social_pink_subtitle_requires_its_font_file(self) -> None:
        profile = self.store.get("subtitle", "social_pink")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(self.root / "empty-local")}, clear=False):
            with self.assertRaisesRegex(ProviderTestError, "字体文件不存在"):
                test_subtitle_profile(profile, self.paths)


class SnapshotHashTests(unittest.TestCase):
    def test_pipeline_input_hash_changes_when_snapshot_changes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root / "script.txt"
            raw = root / "raw"
            raw.mkdir()
            script.write_text("测试脚本", encoding="utf-8")
            snapshot = root / "profile_snapshot.json"
            snapshot.write_text('{"schema_version":1,"sha256":"a"}', encoding="utf-8")
            first = _build_input_hash(
                {"version": 1},
                script,
                raw,
                skip_draft=True,
                draft_root=root / "draft",
                visual_mode="sourced",
            )
            snapshot.write_text('{"schema_version":1,"sha256":"b"}', encoding="utf-8")
            second = _build_input_hash(
                {"version": 1},
                script,
                raw,
                skip_draft=True,
                draft_root=root / "draft",
                visual_mode="sourced",
            )
            self.assertNotEqual(first, second)


class MigrationTests(unittest.TestCase):
    def test_migration_dry_run_never_modifies_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "source"
            target = base / "target"
            (source / ".runtime").mkdir(parents=True)
            (source / ".runtime" / "model.bin").write_bytes(b"model")
            (source / "projects").mkdir()
            with mock.patch.object(studio_migrate, "CODE_ROOT", source):
                report = studio_migrate.migrate_workspace(target, source=source, apply=False)
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue((source / ".runtime" / "model.bin").is_file())
            self.assertFalse(target.exists())

    @unittest.skipUnless(os.name == "nt", "junction migration is a Windows feature")
    def test_locked_outputs_are_preserved_after_verified_copy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "source"
            target = base / "target"
            for name in (".runtime", ".cache", "raw", "outputs", "projects"):
                (source / name).mkdir(parents=True)
            (source / "outputs" / "preview.mp4").write_bytes(b"verified-media")
            original_replace = studio_migrate.os.replace

            def replace_with_locked_outputs(src: object, dst: object) -> None:
                if Path(src).name == "outputs":
                    raise PermissionError("simulated directory handle")
                original_replace(src, dst)

            with (
                mock.patch.object(studio_migrate, "CODE_ROOT", source),
                mock.patch.object(studio_migrate, "_process_running", return_value=False),
                mock.patch.object(studio_migrate, "locking_processes", return_value=[]),
                mock.patch.object(studio_migrate.os, "replace", side_effect=replace_with_locked_outputs),
                mock.patch.object(studio_migrate, "_junction", side_effect=lambda link, destination: link.mkdir()),
                mock.patch.object(studio_migrate, "_is_reparse_point", return_value=False),
                mock.patch.object(studio_migrate, "_draft_media_check", return_value={"references": 0, "missing": 0}),
                mock.patch.object(studio_migrate, "SettingsStore") as settings_store,
            ):
                settings_store.return_value.initialize.return_value = {"workspace": str(target), "jianying_draft_root": str(base / "draft")}
                settings_store.return_value.root = base / "appdata"
                report = studio_migrate.migrate_workspace(target, source=source, apply=True, import_secrets=False)
            self.assertEqual(report["status"], "migrated")
            self.assertTrue((source / "outputs" / "preview.mp4").is_file())
            self.assertTrue((target / "outputs" / "preview.mp4").is_file())
            self.assertIn(str(source / "outputs"), report["preserved_legacy_directories"])


if __name__ == "__main__":
    unittest.main()
