from __future__ import annotations

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


class BuildScanner:
    name = "build"
    priority = 80

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        files_set = {path.name for path in context.files}
        dependencies: dict[str, str] = {}
        confidence = 0.0

        if "uv.lock" in files_set or "pyproject.toml" in files_set:
            dependencies = {"package_manager": "uv", "lock_file": "uv.lock"}
            confidence = 0.9
        if "poetry.lock" in files_set:
            dependencies = {"package_manager": "poetry", "lock_file": "poetry.lock"}
            confidence = 0.9
        if "package-lock.json" in files_set:
            dependencies = {"package_manager": "npm", "lock_file": "package-lock.json"}
            confidence = max(confidence, 0.8)
        if "yarn.lock" in files_set:
            dependencies = {"package_manager": "yarn", "lock_file": "yarn.lock"}
            confidence = max(confidence, 0.8)
        if "pnpm-lock.yaml" in files_set:
            dependencies = {"package_manager": "pnpm", "lock_file": "pnpm-lock.yaml"}
            confidence = max(confidence, 0.8)
        if "go.mod" in files_set:
            dependencies = {"package_manager": "go", "lock_file": "go.sum"}
            confidence = max(confidence, 0.9)
        if "Cargo.toml" in files_set:
            dependencies = {"package_manager": "cargo", "lock_file": "Cargo.lock"}
            confidence = max(confidence, 0.9)
        if "Makefile" in files_set:
            dependencies = {"package_manager": "make", "lock_file": ""}
            confidence = max(confidence, 0.6)

        findings = []
        if dependencies:
            findings.append(
                ScanFinding(
                    key="dependencies",
                    value=dependencies,
                    confidence=confidence,
                    source=self.name,
                )
            )

        suggestions = []
        if not dependencies:
            suggestions.append("No build system detected; configure dependencies manually.")

        return ScanResult(
            name=self.name,
            confidence=confidence,
            findings=findings,
            suggestions=suggestions,
        )
