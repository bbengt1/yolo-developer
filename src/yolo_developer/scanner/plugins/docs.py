from __future__ import annotations

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


class DocsScanner:
    name = "docs"
    priority = 50

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        files = {path.name.lower() for path in context.files}
        directories = {path.as_posix().lower() for path in context.directories}

        has_docs = "docs" in directories
        readmes = [name for name in files if name.startswith("readme")]

        findings = []
        confidence = 0.0
        if readmes or has_docs:
            confidence = 0.8
            findings.append(
                ScanFinding(
                    key="docs",
                    value={"docs_dir": "docs" if has_docs else "", "readmes": readmes},
                    confidence=confidence,
                    source=self.name,
                )
            )
            findings.append(
                ScanFinding(
                    key="conventions",
                    value=["Documentation found"],
                    confidence=0.6,
                    source=self.name,
                )
            )

        return ScanResult(
            name=self.name,
            confidence=confidence,
            findings=findings,
            suggestions=[],
        )
