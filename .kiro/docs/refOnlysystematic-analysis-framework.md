---
inclusion: always
---

# Systematic Analysis Framework for Complex Knowledge Extraction

## Meta-Pattern: Chunked Processing with Multi-Perspective Analysis

This steering document establishes a systematic approach for extracting comprehensive insights from large, complex documentation or content repositories. Based on proven methodologies for strategic intelligence extraction.

## Core Methodology

### 1. Content Segmentation Strategy

**Principle**: Break large content into manageable, overlapping chunks to ensure thorough analysis without cognitive overload.

**Implementation Pattern**:
- Process content in fixed-size chunks (typically 300-500 lines for text, or logical units for code)
- Use 10-20 line overlap between chunks to maintain context continuity
- Track progress systematically with chunk identifiers and source mapping
- Maintain source traceability throughout the process

### 2. Multi-Persona Expert Council Framework

**Principle**: Activate diverse expert perspectives to ensure comprehensive analysis and challenge assumptions.

**Required Personas**:
- **Domain Expert**: Deep technical knowledge of the subject matter
- **Strategic Analyst**: Business and competitive positioning insights
- **Implementation Specialist**: Practical execution and technical feasibility
- **User Experience Advocate**: End-user perspective and workflow optimization
- **Skeptical Engineer/Devil's Advocate**: Challenge assumptions and identify risks (MANDATORY)

**Council Process**:
1. Each persona provides opening analysis
2. Skeptical Engineer challenges primary assertions
3. Other experts respond to challenges
4. Synthesize refined insights into cohesive conclusions

### 3. Conceptual Blending for Innovation

**Principle**: Fuse core concepts with unexpected distant domains to generate novel insights.

**Process**:
- Identify conventional approaches first
- Generate 3+ alternative approaches using conceptual blending
- Blend with distant domains (e.g., biology, physics, economics, psychology)
- Evaluate and select most promising hybrid approach
- Justify selection with clear reasoning

### 4. Verification and Quality Assurance

**Principle**: Generate fact-checkable questions to validate major claims and insights.

**Implementation**:
- Generate 5-10 specific, verifiable questions per major insight
- Answer verification questions based on available evidence
- Identify and correct inconsistencies or weaknesses
- Ensure claims are grounded and actionable

## Task Breakdown Patterns

### Meta-Pattern: Hierarchical Task Organization for Complex Projects

**Principle**: Maintain granular task detail while reducing cognitive load through strategic grouping.

**Problem Solved**: Large projects with hundreds of granular tasks become overwhelming to navigate and execute, leading to analysis paralysis and poor progress tracking.

**Solution Framework**:
- **Preserve Granularity**: Keep all detailed sub-tasks for thorough execution
- **Reduce High-Level Buttons**: Group related tasks under fewer main categories
- **Logical Organization**: Use natural groupings (alphabetical, functional, chronological)
- **Clear Hierarchy**: Main tasks contain numbered sub-tasks for easy navigation

**Implementation Pattern**:
```
- [ ] Main Task Group (X items)
  - [ ] X.1 Specific granular task
  - [ ] X.2 Another specific granular task
  - [ ] X.3 Final specific granular task
```

**Example Application - File Processing**:
```
- [ ] 2. Process A Files (5 files)
  - [ ] 2.1 Analyze A01-README-MOSTIMP.txt
  - [ ] 2.2 Analyze A01Rust300Doc20250923.docx.md
  - [ ] 2.3 Analyze A01Rust300Doc20250923.docx.txt
  - [ ] 2.4 Analyze Aadeesh Patni.txt
  - [ ] 2.5 Analyze abc.txt
```

**Benefits**:
- **Reduced Overwhelm**: 17 main buttons instead of 93+ individual tasks
- **Maintained Detail**: Every granular task preserved for systematic execution
- **Progress Clarity**: Clear completion tracking at both levels
- **Logical Flow**: Natural groupings make execution order obvious
- **Scalability**: Pattern works for projects of any size

**When to Apply**:
- Projects with 50+ granular tasks
- Natural groupings exist (alphabetical, functional, temporal)
- Need to balance thoroughness with usability
- Multiple stakeholders need different levels of detail

### Pattern 1: Systematic Processing Tasks

**Structure**: Break large processing jobs into measurable, trackable units

**Example Format**:
```
- [ ] Process [Source] chunks X-Y (lines A-B)
  - Apply analytical framework to each chunk
  - Extract [specific insight types]
  - Maintain [quality standards]
  - Cross-reference with [previous findings]
  - _Requirements: [specific requirement references]_
```

**Key Characteristics**:
- Specific line ranges or logical boundaries
- Clear deliverables for each chunk
- Progress tracking built into task structure
- Quality assurance steps included
- Requirement traceability maintained

### Pattern 2: Synthesis and Organization Tasks

**Structure**: Transform extracted insights into structured, actionable formats

**Example Format**:
```
- [ ] Categorize [insights] by [classification system]
  - Organize by [primary dimension] and [secondary dimension]
  - Ensure each [item] includes [required elements]
  - Cross-reference [related items] and identify [dependencies]
  - _Requirements: [specific requirement references]_
```

**Key Characteristics**:
- Clear categorization criteria
- Required elements specified for each item
- Cross-referencing and relationship mapping
- Dependency identification

### Pattern 3: Quality Assurance Tasks

**Structure**: Validate completeness, consistency, and quality of extracted insights

**Example Format**:
```
- [ ] Verify [aspect] of [deliverable]
  - Confirm [specific quality criteria]
  - Validate [consistency requirements]
  - Ensure [completeness standards]
  - Document [traceability requirements]
  - _Requirements: [specific requirement references]_
```

## Application Guidelines

### When to Use This Framework

- **Large Documentation Analysis**: Processing extensive technical documentation, specifications, or advisory materials
- **Strategic Intelligence Extraction**: Extracting actionable insights from complex source materials
- **Multi-Source Synthesis**: Combining insights from multiple documents or data sources
- **Comprehensive Research Projects**: Ensuring thorough coverage of complex subject matter

### Task Granularity Principles

1. **Measurable Progress**: Each task should represent 2-4 hours of focused work
2. **Clear Deliverables**: Specific outputs that can be verified and validated
3. **Incremental Value**: Each task builds on previous work and adds measurable value
4. **Quality Gates**: Built-in verification and quality assurance steps
5. **Requirement Traceability**: Clear links to specific requirements or objectives

### Progress Tracking Standards

- **Completion Metrics**: Quantifiable measures of progress (chunks processed, insights extracted)
- **Quality Indicators**: Verification questions answered, cross-references validated
- **Source Mapping**: Traceability from insights back to source material
- **Milestone Markers**: Clear checkpoints for major phases of work

## Quality Standards

### Analysis Rigor
- Apply systematic methodology to every content unit
- Maintain consistent analytical standards throughout
- Document reasoning and decision-making process
- Validate insights through multiple perspectives

### Output Quality
- Ensure actionable and implementable recommendations
- Maintain logical consistency across all insights
- Provide sufficient detail for implementation
- Include success metrics and validation criteria

### Documentation Standards
- Clear source attribution for all insights
- Structured format for easy navigation and reference
- Cross-references between related concepts
- Progress tracking and completion verification

This framework ensures comprehensive, rigorous analysis while maintaining systematic progress tracking and quality assurance throughout complex knowledge extraction projects.


# Use Folder analysis scripts to check your work and progress 

Use folder analysis script such as /home/amuldotexe/Desktop/Game20250926/number-12-grimmauld-place/scripts/tree-with-wc.sh to constantly check your progress -and you complete a file make a simple git commit with some details on the progress

=================================

.
├── A01-README-MOSTIMP.txt
├── A01Rust300Doc20250923.docx.md
├── A01Rust300Doc20250923.docx.txt
├── abc.txt
├── Bullet-Proof Mermaid Prompts_ Square-Perfect Diagrams from Any LLM.txt
├── DeconstructDeb_trun_c928898c8ef7483eadc3541123e5d88f.txt
├── DeconstructDebZero-Trust.deb Dissection_ A Rust Toolchain for Safe, Deep-Dive Package Analysis.txt
├── design.txt
├── Evaluating OSS Rust Ideas.md
├── Fearless & Fast_ 40+ Proven Rayon Idioms that Slash Bugs and Unlock Core-Level Speed in Rust.txt
├── FINAL_DELIVERABLES_SUMMARY.txt
├── From Zero to Constitutional Flowcharts_ Fast-Track, Risk-Free Paths with LLMs.txt
├── jules20250926.txt
├── LIMITATIONS_AND_ADVANCED_TECHNIQUES.txt
├── Mermaid_trun_c928898c8ef7483eb8257cb7dc52ac9a.json
├── MSFT C SUITE trun_8a68e63f9ca64238a77c8282312e719a.json
├── OpenSearch Contribution and Innovation Ideas.txt
├── Padé Approximations_ PMF and Build_.txt
├── PARSELTONGUE_BEST_PRACTICES_GUIDE.txt
├── PARSELTONGUE_V2_RECOMMENDATIONS.txt
├── PRDsRust300p1.txt
├── README.txt
├── Reference Conversation.txt
├── Researchv1..txt
├── Rust30020250815_complete.txt
├── Rust30020250815_full.txt
├── Rust30020250815_minto.txt
├── Rust30020250815.txt
├── Rust300AB20250926.md
├── Rust300 Consolidated Pre-Development Specification for Minimalist Rust Utilities.txt
├── Rust300 Rust CPU Library Idea Generation.txt
├── Rust300 Rust Library Idea Generation.txt
├── Rust300 Rust Micro-Libraries for CPU-Intensive Tasks.txt
├── Rust300 Rust Micro-Library Idea Generation.txt
├── Rust300 Rust OSS Project Planning.txt
├── Rust Clippy Playbook_ 750 Proven Idioms That Slash Bugs & Boost Speed.txt
├── rust_complexity_quick_reference.txt
├── RustGringotts High PMF 20250924.md
├── Rust Idiomatic Patterns Deep Dive_.txt
├── RustJobs Adoption Data Expansion & Analysis.txt
├── RustJobs Rust Adoption_ Job Market Analysis.txt
├── Rust Library Ideas_ Criteria Analysis.md
├── RustLLM Rust300 Rust OSS Project Planning.txt
├── Rust LLM Rust Micro-Library Ideas Search_.txt
├── RustLLM trun_4122b840faa84ad78124aa70192d96ab.json
├── RustLLM trun_4122b840faa84ad79c9c39b3ebabf8a0.json
├── RustLLM trun_4122b840faa84ad7bd3793df0e5f39ee(1).txt
├── Rust Micro-Libraries for CPU.txt
├── Rust Micro-Library Ideas Search_.txt
├── Rust OSS Contribution and Hiring.txt
├── Rust Patterns List.txt
├── Shared Research - Parallel Web Systems, Inc..txt
├── tasks.txt
├── task-tracker.txt
├── tokio-rs-axum-8a5edab282632443.txt
├── Tokio’s 20%_ High-Leverage Idioms that Eliminate Bugs and Turbo-Charge Rust Async Apps.txt
├── trun_1b986480e1c84d75a6ad29b1d72efff6.json
├── trun_1b986480e1c84d75b02b7fba69f359c9.json
├── trun_1b986480e1c84d75bc94381ba6d21189.json
├── trun_82b88932a051498485c362bd64070533.json
├── trun_82b88932a0514984938aec7b95fbee66.json
├── trun_82b88932a0514984a4fd517f37b144be.json
├── trun_82b88932a0514984bbc73cb821649c97.json
├── trun_82b88932a0514984bc2d6d98eab7423f.json
├── trun_c30434831bfd40abb830834705a1c6c4.json
├── trun_c928898c8ef7483e86b41b8fea65209e.txt
├── trun_c928898c8ef7483e893944f08945f3a3.txt
├── trun_c928898c8ef7483ea7128f70251c9860.txt
├── trun_c928898c8ef7483eb1a233d6dc8501f8.txt
├── trun_d3115feeb76d407d8a22aec5ca6ffa26.json
├── trun_d3115feeb76d407d8d2e6a5293afb28d.json
├── trun_d3115feeb76d407db7f7be20d7602124.json
├── trun_d3115feeb76d407dbe3a09f93b0d880d.json
├── trun_da5838edb25d44d389074277f64aa5e8.json
├── trun_da5838edb25d44d38ae43a28e5428fa3.json
├── trun_da5838edb25d44d39eabe0c3e214baf8.json
├── trun_da5838edb25d44d3a70374acaa5842fc.json
├── trun_da5838edb25d44d3aafd38d1d60f89ec.json
├── trun_da5838edb25d44d3b54fe7c1fd3e5d2a.json
├── trun_f92ce0b9ccf145868312b54196c93066.json
├── trun_f92ce0b9ccf14586858c7f9a1b1c4e31.json
├── trun_f92ce0b9ccf1458685ef2c96c371a704.json
├── trun_f92ce0b9ccf1458688f3b22f0aca35d5.json
├── trun_f92ce0b9ccf14586aa356591292c19b9.json
├── trun_f92ce0b9ccf14586afada492fcd8d658.json
├── trun_f92ce0b9ccf14586b5f5c6afe0dd8945.json
├── trun_f92ce0b9ccf14586b67676d6d94d7362.json
├── trun_f92ce0b9ccf14586bc02b7d9ef19971d.json
├── UnpackKiro_trun_c928898c8ef7483eace3078d9b2f944e.txt
├── UnpackKiro_Unpack With Confidence_ A Secure, Streaming-Fast Deep-Dive into Kiro’s.deb.txt
├── use-case-202509 (1).txt
├── workflow_patterns.txt
└── WORKFLOW_TEMPLATES.txt

0 directories, 94 files