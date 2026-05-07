#!/usr/bin/env python3
"""
Skill Packager - Creates a distributable .skill file of a skill folder

Usage:
    python utils/package_skill.py <path/to/skill-folder> [output-directory]

Example:
    python utils/package_skill.py skills/public/my-skill
    python utils/package_skill.py skills/public/my-skill ./dist
"""

import sys
import logging
import zipfile
from pathlib import Path
from quick_validate import validate_skill

logger = logging.getLogger(__name__)


def package_skill(skill_path, output_dir=None):
    """
    Package a skill folder into a .skill file.

    Args:
        skill_path: Path to the skill folder
        output_dir: Optional output directory for the .skill file (defaults to current directory)

    Returns:
        Path to the created .skill file, or None if error
    """
    skill_path = Path(skill_path).resolve()

    # Validate skill folder exists
    if not skill_path.exists():
        logger.error("Error: Skill folder not found: %s", skill_path)
        return None

    if not skill_path.is_dir():
        logger.error("Error: Path is not a directory: %s", skill_path)
        return None

    # Validate SKILL.md exists
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        logger.error("Error: SKILL.md not found in %s", skill_path)
        return None

    # Run validation before packaging
    logger.info("Validating skill...")
    valid, message = validate_skill(skill_path)
    if not valid:
        logger.error("Validation failed: %s", message)
        logger.error("Please fix the validation errors before packaging.")
        return None
    logger.info("%s", message)

    # Determine output location
    skill_name = skill_path.name
    if output_dir:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    skill_filename = output_path / f"{skill_name}.skill"

    # Create the .skill file (zip format)
    try:
        with zipfile.ZipFile(skill_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the skill directory
            for file_path in skill_path.rglob('*'):
                if file_path.is_file():
                    # Calculate the relative path within the zip
                    arcname = file_path.relative_to(skill_path.parent)
                    zipf.write(file_path, arcname)
                    logger.info("Added: %s", arcname)

        logger.info("Successfully packaged skill to: %s", skill_filename)
        return skill_filename

    except Exception as e:
        logger.error("Error creating .skill file: %s", e)
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        logger.error(
            "Usage: python utils/package_skill.py <path/to/skill-folder> [output-directory]"
        )
        logger.error("Example:")
        logger.error("  python utils/package_skill.py skills/public/my-skill")
        logger.error("  python utils/package_skill.py skills/public/my-skill ./dist")
        sys.exit(1)

    skill_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info("Packaging skill: %s", skill_path)
    if output_dir:
        logger.info("Output directory: %s", output_dir)

    result = package_skill(skill_path, output_dir)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
