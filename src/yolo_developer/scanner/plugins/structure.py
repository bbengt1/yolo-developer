from __future__ import annotations

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


class StructureScanner:
    name = "structure"
    priority = 60

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        directories = {path.as_posix() for path in context.directories}

        source_root = "src" if "src" in directories else "lib" if "lib" in directories else "."
        test_root = (
            "tests"
            if "tests" in directories
            else "__tests__" if "__tests__" in directories else ""
        )
        config_root = "config" if "config" in directories else ""

        structure = {
            "source_root": source_root,
            "test_root": test_root,
            "config_root": config_root,
        }

        confidence = 0.7
        findings = [
            ScanFinding(
                key="structure",
                value=structure,
                confidence=confidence,
                source=self.name,
            )
        ]

        suggestions = []
        if not test_root:
            suggestions.append("No test directory detected; confirm test layout.")

        return ScanResult(
            name=self.name,
            confidence=confidence,
            findings=findings,
            suggestions=suggestions,
        )
