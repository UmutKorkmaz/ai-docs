# Chapter 4: Early AI Approaches (1950s-1970s)

> **Prerequisites**: [Birth of AI](03_Birth_of_AI.md)
>
> **Learning Objectives**:
> - Understand the symbolic AI paradigm that dominated early research
> - Explore early attempts at natural language processing and computer vision
> - Learn about the first AI winter and its causes
>
> **Related Topics**: [Mathematical Foundations](05_Mathematical_Foundations.md) | [Computational Theory](06_Computational_Theory.md)

## Symbolic AI and Logic-Based Systems

The period from the 1950s to 1970s was dominated by symbolic AI, also known as "Good Old-Fashioned AI" (GOFAI). This approach was based on the idea that intelligence could be achieved through the manipulation of symbols according to formal rules.

### The Symbolic AI Paradigm

**Physical Symbol System Hypothesis**
- Proposed by Newell and Simon in 1976
- States that "a physical symbol system has the necessary and sufficient means for general intelligent action"
- Assumes intelligence operates through manipulation of symbols
- Claims that any intelligent system must be a symbol system
- Provided theoretical foundation for symbolic AI

**Core Principles**
- **Symbol Manipulation**: Intelligence emerges from symbol processing
- **Formal Logic**: Mathematical reasoning as basis for intelligence
- **Knowledge Representation**: Encoding facts and rules
- **Search Strategies**: Finding solutions through systematic exploration
- **Problem Decomposition**: Breaking complex problems into smaller ones

**Research Methodology**
- Top-down approach to intelligence
- Emphasis on reasoning and problem-solving
- Focus on high-level cognitive processes
- Use of formal methods and mathematical rigor
- Development of general problem-solving architectures

### Key Symbolic AI Approaches

#### Production Systems

**IF-THEN Rule Architecture**
- Production rules: IF condition THEN action
- Working memory: Current state of knowledge
- Rule interpreter: Applies rules to working memory
- Conflict resolution: Chooses among applicable rules

**Applications and Examples**
- **DENDRAL** (1965): First expert system for organic chemistry
  - Identified molecular structures from mass spectrometry data
  - Used heuristic search and rule-based reasoning
  - Significantly outperformed human chemists
  - Demonstrated practical value of expert systems

**Technical Implementation**
```lisp
; Example production rule from DENDRAL
(IF (and (molecule-has-carbon C1)
         (molecule-has-carbon C2)
         (bond-between C1 C2 double))
    THEN (suggest-alkene-structure))
```

#### Logic Programming

**Formal Logic for AI**
- First-order predicate calculus as knowledge representation
- Resolution theorem proving for inference
- Unification for pattern matching
- Backward chaining for goal-directed reasoning

**Prolog Language (1972)**
- Developed by Alain Colmerauer and Robert Kowalski
- First major logic programming language
- Automated theorem proving capabilities
- Used for natural language processing and expert systems

**Sample Prolog Program**
```prolog
% Family relationships in Prolog
parent(john, mary).
parent(john, bob).
parent(mary, alice).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% Query: grandparent(john, X)
% Result: X = alice
```

#### Semantic Networks

**Graph-Based Knowledge Representation**
- Nodes represent concepts or objects
- Edges represent relationships between concepts
- Hierarchical organization of knowledge
- Inheritance of properties through relationships

**Early Semantic Network Systems**
- **QUILLAN's TLC (1966)**: Teachable Language Comprehender
- **Norman & Rumelhart's LNR (1975)**: Limited Natural Reasoner
- **Schank's Conceptual Dependency (1972)**: Meaning representation

**Example Semantic Network**
```
          [Bird]
            |
       [is-a]
            |
          [Robin]
         /    |    \
  [has-color] [can]  [has-size]
        |       |        |
      [Red]  [Fly]   [Small]
```

#### Frame-Based Systems

**Structured Knowledge Representation**
- Frames represent stereotyped situations or objects
- Slots contain attributes and values
- Default values and inheritance
- Procedures attached to slots (demons)

**Marvin Minsky's Frames (1975)**
- Proposed as alternative to semantic networks
- Emphasis on structured knowledge organization
- Support for default reasoning
- Integration of procedural and declarative knowledge

**Example Frame Structure**
```
Frame: BIRD
  Slots:
    is-a: ANIMAL
    can-fly: YES (default)
    has-wings: YES
    has-feathers: YES
    size: SMALL (default)

Frame: PENGUIN
  Slots:
    is-a: BIRD
    can-fly: NO (override)
    size: MEDIUM (override)
```

### Early Successes of Symbolic AI

#### DENDRAL (1965)

**Achievements**
- First expert system to achieve practical success
- Outperformed human experts in specific domain
- Combined heuristic search with chemical knowledge
- Demonstrated value of knowledge engineering

**Technical Innovation**
- Used generate-and-test methodology
- Employed constraint satisfaction
- Integrated domain-specific heuristics
- Established expert system paradigm

#### MYCIN (1970s)

**Medical Diagnosis System**
- Developed at Stanford University
- Diagnosed blood infections and recommended treatments
- Achieved 69% accuracy compared to human experts
- Influenced medical decision support systems

**Knowledge Base Design**
- Rules represented as certainty factors
- Backward chaining for diagnosis
- Explanation capabilities
- User interaction and consultation

**Sample MYCIN Rule**
```
IF the infection is meningitis
AND the patient is an adult
AND the patient has a compromised immune system
THEN there is suggestive evidence (0.6) that the infection is fungal
```

#### SHRDLU (1970)

**Natural Language Understanding**
- Created by Terry Winograd at MIT
- Operated in a "blocks world" environment
- Understood natural language commands
- Demonstrated integration of language and reasoning

**Technical Features**
- Used procedural semantics for language understanding
- Integrated planning and execution
- Maintained world model and beliefs
- Provided explanations of reasoning

**Sample Interaction**
```
User: Put the small pyramid on the large block.
System: OK.
User: What did the pyramid touch before you moved it?
System: The table.
User: Can a pyramid be supported by a block?
System: Yes, if the block is large enough.
```

## Early Natural Language Processing

The 1960s and 1970s saw significant advances in natural language processing, driven by both theoretical developments and practical applications.

### Machine Translation (1954)

**Georgetown-IBM Experiment**
- First public demonstration of machine translation
- Translated Russian sentences to English
- Limited vocabulary (250 words) but impressive demonstration
- Generated enthusiasm for automated translation

**Technical Approach**
- Used bilingual dictionary lookup
- Applied simple grammatical rules
- Hand-coded translation patterns
- Limited success with complex sentences

**Challenges Encountered**
- Syntactic ambiguity in natural language
- Multiple meanings of words
- Context-dependent interpretation
- Cultural and idiomatic expressions

### Syntactic Theories

#### Noam Chomsky's Transformational Grammar (1957)

**Revolutionary Impact**
- Revolutionized linguistics and NLP
- Distinguished deep structure from surface structure
- Proposed universal grammar
- Influenced computational linguistics

**Key Concepts**
- **Phrase Structure Rules**: Generate basic sentence structures
- **Transformational Rules**: Modify basic structures
- **Deep Structure**: Underlying meaning representation
- **Surface Structure**: Actual sentence form

**Example Transformation**
```
Deep Structure: [The boy] [see] [the girl]
Transformation: Passive Voice
Surface Structure: [The girl] [is seen] [by the boy]
```

#### Context-Free Grammars

**Formal Grammar Framework**
- Rules for generating valid sentences
- Parsing algorithms for analysis
- Foundation for many NLP systems
- Balance between expressiveness and computability

**CFG Example**
```
S → NP VP
NP → Det N | Det Adj N
VP → V | V NP
Det → the | a
N → boy | girl | ball
V → throws | catches
Adj → red | big
```

#### Augmented Transition Networks (ATN)

**Enhanced Parsing Capability**
- Extended finite-state automata
- Added conditions and actions
- Support for complex grammatical features
- Used in many early NLP systems

**ATN Features**
- Recursive grammar rules
- Feature passing between nodes
- Agreement checking
- Semantic interpretation

### Early NLP Systems

#### SHRDLU

**Blocks World Understanding**
- Combined language understanding with physical reasoning
- Used procedural semantics
- Maintained dynamic world model
- Demonstrated integrated AI system

#### LUNAR

**Scientific Question Answering**
- Answered questions about moon rocks
- Used database of Apollo mission data
- Applied ATN parsing technology
- Demonstrated practical NLP application

#### MARGIE

**Meaning Representation**
- Developed at Yale University
- Used Conceptual Dependency theory
- Focused on semantic interpretation
- Explored paraphrase generation

## Computer Vision Beginnings

Early computer vision research dramatically underestimated the complexity of visual perception but laid important groundwork for modern approaches.

### The Summer Vision Project (1966)

**Ambitious Goal**
- Marvin Minsky's project to solve vision in one summer
- Assigned to an undergraduate student
- Goal was to have computer describe scenes from images
- Dramatically underestimated the complexity of vision

**Technical Challenges**
- Edge detection and segmentation
- Object recognition and classification
- 3D reconstruction from 2D images
- Handling lighting and shadow variations
- Dealing with occlusion and partial views

**Long-term Impact**
- Established vision as legitimate AI research area
- Generated interest in computational perception
- Led to more realistic research agendas
- Influenced development of computer vision field

### Early Vision Approaches

#### Edge Detection

**Finding Boundaries**
- Identifying intensity discontinuities
- Gradient-based methods
- Template matching approaches
- Thresholding techniques

**Key Algorithms**
- **Roberts Cross (1965)**: Simple edge detection
- **Sobel Operator (1968)**: Improved gradient estimation
- **Prewitt Operator**: Edge detection with noise reduction
- **Canny Edge Detector (1986)**: Optimal edge detection

#### Line Drawing Analysis

**2D Interpretation**
- Interpreting simple line drawings
- Junction and vertex analysis
- Constraint satisfaction for scene understanding
- Handling line ambiguity

**Labeling Schemes**
- Huffman-Clowes labeling
- Waltz algorithm for line drawing interpretation
- Relaxation labeling techniques
- Consistency checking for scene interpretation

#### 3D Reconstruction

**From 2D to 3D**
- Building 3D models from 2D images
- Shape from shading techniques
- Stereo vision approaches
- Structure from motion methods

**Challenges**
- Ambiguity in depth perception
- Handling occlusion and transparency
- Recovering surface properties
- Integrating multiple views

### Shakey the Robot (1969-1972)

**First Mobile Robot with Vision**
- Developed at Stanford Research Institute
- Combined multiple sensors and reasoning
- Demonstrated early AI integration
- Pioneered robot navigation

**Technical Features**
- TV camera for vision
- Range finder for distance measurement
- Onboard computer for processing
- Radio link to mainframe computer

**Navigation Architecture**
1. **Perception**: Process sensor data
2. **Model Building**: Create world representation
3. **Planning**: Generate action sequences
4. **Execution**: Control robot movements
5. **Monitoring**: Update world model

**Achievements**
- Navigate complex indoor environments
- Push objects to goal locations
- Avoid obstacles dynamically
- Demonstrate integrated AI system

## The First AI Winter and Its Causes

The period from 1974 to 1980 marked the first "AI Winter," a time of reduced funding, diminished enthusiasm, and critical reassessment of AI research directions.

### Reasons for AI Winter (1974-1980)

#### 1. Overpromising and Underdelivering

**Unrealistic Expectations**
- Researchers made grand claims that couldn't be fulfilled
- Media created unrealistic expectations
- Predictions of human-level AI within decades
- Demonstrations in limited domains misrepresented as general intelligence

**Communication Issues**
- Technical limitations not well communicated
- Success stories exaggerated
- Failures and difficulties downplayed
- Gap between research and public understanding

#### 2. Computational Limitations

**Hardware Constraints**
- Computers were too slow for complex AI tasks
- Memory limitations restricted problem size
- Storage capacity limited knowledge bases
- Processing power insufficient for real-time applications

**Software Challenges**
- Programming languages inadequate for AI
- Lack of efficient algorithms
- Difficulty in scaling systems
- Integration problems between components

#### 3. Combinatorial Explosion

**Search Space Complexity**
- Search spaces grew exponentially with problem size
- Heuristics couldn't overcome fundamental complexity
- Brute-force approaches infeasible for real problems
- Worst-case performance was unacceptable

**Examples**
- Chess: 10^120 possible game states
- Natural language: infinite possible sentences
- Vision: infinite image variations
- Planning: exponential action sequences

#### 4. Lack of Data

**Data Scarcity**
- No large datasets for training
- Manual rule creation was time-consuming
- Difficulty in acquiring domain knowledge
- Limited availability of test cases

**Knowledge Acquisition Bottleneck**
- Expert knowledge hard to extract
- Domain experts expensive and scarce
- Knowledge inconsistent and incomplete
- Validation and verification difficult

#### 5. Funding Cuts

**British Lighthill Report (1973)**
- Sir James Lighthill reviewed AI progress
- Criticized lack of fundamental advances
- Questioned practical applications
- Recommended reduced funding

**DARPA Funding Reduction**
- U.S. Defense Advanced Research Projects Agency reduced AI funding
- Shifted funding to more practical projects
- Emphasis on short-term results
- Reduced support for basic research

**Other Funding Sources**
- Industry investment decreased
- Academic budgets tightened
- Government priorities shifted
- Economic recession affected research funding

### Impact of the Winter

#### Research Community Effects

**Lab Closures and Redirection**
- Many AI labs closed or redirected research
- Researchers moved to other fields
- Some universities reduced AI programs
- Industry research scaled back

**Terminology Changes**
- Term "AI" became unpopular
- "Expert systems" and "knowledge engineering" preferred
- "Applied AI" distinguished from "pure AI"
- Emphasis on practical applications

#### Research Direction Changes

**More Realistic Goals**
- Focus on narrow, achievable tasks
- Emphasis on practical applications
- Integration with existing systems
- Commercial potential considered

**Increased Rigor**
- Better experimental methodology
- More thorough evaluation
- Clearer success criteria
- Honest assessment of limitations

#### Long-term Positive Effects

**Maturation of Field**
- More realistic expectations
- Better understanding of challenges
- Improved research methodology
- Stronger theoretical foundations

**Foundation for Revival**
- Expert systems emerged as practical application
- Connectionist approaches gained attention
- Better understanding of AI limitations
- Preparation for future advances

## Legacy and Lessons Learned

### Technical Contributions

**Foundational Algorithms**
- Many early techniques still used today
- Search algorithms remain fundamental
- Knowledge representation methods evolved
- Planning techniques improved and refined

**Software Tools**
- LISP and Prolog influenced language design
- Development environments for AI research
- Knowledge engineering methodologies
- Testing and evaluation frameworks

### Methodological Advances

**Research Methods**
- Better experimental design
- More rigorous evaluation
- Clearer success criteria
- Improved documentation

**Project Management**
- More realistic planning
- Better resource allocation
- Clearer milestone definition
- Risk assessment and mitigation

### Philosophical Insights

**Nature of Intelligence**
- Deeper understanding of intelligence complexity
- Recognition of multiple forms of intelligence
- Appreciation of embodied cognition
- Understanding of learning requirements

**Human-Machine Differences**
- Recognition of different cognitive strengths
- Understanding of complementary capabilities
- Appreciation of human expertise
- Recognition of interaction challenges

### Transition to Modern AI

**From Symbolic to Statistical**
- Gradual shift toward data-driven approaches
- Integration of learning and reasoning
- Emphasis on probabilistic methods
- Combination of multiple paradigms

**Foundation for Future Success**
- Lessons learned from failures
- Better understanding of requirements
- Improved research methodologies
- Realistic expectations and goals

---

**Next Section**: [Mathematical Foundations](05_Mathematical_Foundations.md) - The mathematical tools underlying AI systems

**Related Topics**: [Computational Theory](06_Computational_Theory.md) | [Cognitive Science Foundations](07_Cognitive_Science_Foundations.md)

**Historical Context**: See [Timeline](D_Timeline.md) for AI Winter chronology and recovery