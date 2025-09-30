#!/bin/bash
# Version management script for code-ingest
# Handles semantic versioning and updates all relevant files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
CARGO_MANIFEST="code-ingest/Cargo.toml"
CHANGELOG="code-ingest/CHANGELOG.md"

# Get current version from Cargo.toml
get_current_version() {
    grep '^version = ' "$CARGO_MANIFEST" | sed 's/version = "\(.*\)"/\1/'
}

# Parse semantic version
parse_version() {
    local version=$1
    local regex="^([0-9]+)\.([0-9]+)\.([0-9]+)(-([0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*))?(\+([0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*))?$"
    
    if [[ $version =~ $regex ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        MINOR="${BASH_REMATCH[2]}"
        PATCH="${BASH_REMATCH[3]}"
        PRERELEASE="${BASH_REMATCH[5]}"
        BUILD="${BASH_REMATCH[8]}"
        return 0
    else
        return 1
    fi
}

# Increment version
increment_version() {
    local current_version=$1
    local bump_type=$2
    
    if ! parse_version "$current_version"; then
        print_error "Invalid version format: $current_version"
        exit 1
    fi
    
    case $bump_type in
        "major")
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
        "minor")
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        "patch")
            PATCH=$((PATCH + 1))
            ;;
        *)
            print_error "Invalid bump type: $bump_type (use major, minor, or patch)"
            exit 1
            ;;
    esac
    
    echo "${MAJOR}.${MINOR}.${PATCH}"
}

# Update version in Cargo.toml
update_cargo_version() {
    local new_version=$1
    
    print_status "Updating version in $CARGO_MANIFEST to $new_version"
    
    # Use sed to update the version line
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed
        sed -i '' "s/^version = \".*\"/version = \"$new_version\"/" "$CARGO_MANIFEST"
    else
        # GNU sed
        sed -i "s/^version = \".*\"/version = \"$new_version\"/" "$CARGO_MANIFEST"
    fi
    
    print_success "Updated $CARGO_MANIFEST"
}

# Update changelog
update_changelog() {
    local new_version=$1
    local current_date=$(date +"%Y-%m-%d")
    
    print_status "Updating changelog for version $new_version"
    
    if [[ ! -f "$CHANGELOG" ]]; then
        print_warning "Changelog not found, creating new one"
        cat > "$CHANGELOG" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [$new_version] - $current_date

### Added
- Initial release

EOF
        print_success "Created new changelog"
        return 0
    fi
    
    # Create temporary file for new changelog content
    local temp_changelog=$(mktemp)
    
    # Read the existing changelog and insert new version
    local in_unreleased=false
    local added_new_version=false
    
    while IFS= read -r line; do
        if [[ "$line" == "## [Unreleased]" ]]; then
            echo "$line" >> "$temp_changelog"
            echo "" >> "$temp_changelog"
            echo "## [$new_version] - $current_date" >> "$temp_changelog"
            echo "" >> "$temp_changelog"
            echo "### Added" >> "$temp_changelog"
            echo "- " >> "$temp_changelog"
            echo "" >> "$temp_changelog"
            echo "### Changed" >> "$temp_changelog"
            echo "- " >> "$temp_changelog"
            echo "" >> "$temp_changelog"
            echo "### Fixed" >> "$temp_changelog"
            echo "- " >> "$temp_changelog"
            echo "" >> "$temp_changelog"
            added_new_version=true
        else
            echo "$line" >> "$temp_changelog"
        fi
    done < "$CHANGELOG"
    
    # If we didn't find an [Unreleased] section, add the new version at the top
    if [[ "$added_new_version" == false ]]; then
        local temp_changelog2=$(mktemp)
        echo "# Changelog" > "$temp_changelog2"
        echo "" >> "$temp_changelog2"
        echo "All notable changes to this project will be documented in this file." >> "$temp_changelog2"
        echo "" >> "$temp_changelog2"
        echo "## [Unreleased]" >> "$temp_changelog2"
        echo "" >> "$temp_changelog2"
        echo "## [$new_version] - $current_date" >> "$temp_changelog2"
        echo "" >> "$temp_changelog2"
        echo "### Added" >> "$temp_changelog2"
        echo "- " >> "$temp_changelog2"
        echo "" >> "$temp_changelog2"
        tail -n +1 "$temp_changelog" >> "$temp_changelog2"
        mv "$temp_changelog2" "$temp_changelog"
    fi
    
    # Replace the original changelog
    mv "$temp_changelog" "$CHANGELOG"
    
    print_success "Updated changelog"
    print_warning "Please edit $CHANGELOG to add release notes for version $new_version"
}

# Create git tag
create_git_tag() {
    local version=$1
    local tag_name="v$version"
    
    print_status "Creating git tag: $tag_name"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_warning "Not in a git repository, skipping tag creation"
        return 0
    fi
    
    # Check if tag already exists
    if git tag -l | grep -q "^$tag_name$"; then
        print_warning "Tag $tag_name already exists"
        return 0
    fi
    
    # Create annotated tag
    git tag -a "$tag_name" -m "Release version $version"
    print_success "Created git tag: $tag_name"
    
    print_status "To push the tag, run: git push origin $tag_name"
}

# Validate version format
validate_version() {
    local version=$1
    
    if ! parse_version "$version"; then
        print_error "Invalid version format: $version"
        print_error "Version must follow semantic versioning (e.g., 1.2.3)"
        exit 1
    fi
}

# Show current version
show_current_version() {
    local current_version=$(get_current_version)
    echo "Current version: $current_version"
}

# Show help
show_help() {
    echo "Version management script for code-ingest"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  current                 Show current version"
    echo "  bump <type>            Bump version (major, minor, patch)"
    echo "  set <version>          Set specific version"
    echo "  tag                    Create git tag for current version"
    echo "  help                   Show this help message"
    echo
    echo "Examples:"
    echo "  $0 current             # Show current version"
    echo "  $0 bump patch          # Bump patch version (1.0.0 -> 1.0.1)"
    echo "  $0 bump minor          # Bump minor version (1.0.0 -> 1.1.0)"
    echo "  $0 bump major          # Bump major version (1.0.0 -> 2.0.0)"
    echo "  $0 set 1.2.3           # Set version to 1.2.3"
    echo "  $0 tag                 # Create git tag for current version"
    echo
}

# Main function
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command=$1
    shift
    
    case $command in
        "current")
            show_current_version
            ;;
        "bump")
            if [[ $# -ne 1 ]]; then
                print_error "Bump command requires a type (major, minor, patch)"
                exit 1
            fi
            
            local bump_type=$1
            local current_version=$(get_current_version)
            local new_version=$(increment_version "$current_version" "$bump_type")
            
            print_status "Bumping version from $current_version to $new_version"
            
            update_cargo_version "$new_version"
            update_changelog "$new_version"
            
            print_success "Version bumped to $new_version"
            print_status "Don't forget to:"
            print_status "1. Edit the changelog to add release notes"
            print_status "2. Commit the changes"
            print_status "3. Run '$0 tag' to create a git tag"
            ;;
        "set")
            if [[ $# -ne 1 ]]; then
                print_error "Set command requires a version"
                exit 1
            fi
            
            local new_version=$1
            validate_version "$new_version"
            
            local current_version=$(get_current_version)
            print_status "Setting version from $current_version to $new_version"
            
            update_cargo_version "$new_version"
            update_changelog "$new_version"
            
            print_success "Version set to $new_version"
            ;;
        "tag")
            local current_version=$(get_current_version)
            create_git_tag "$current_version"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"