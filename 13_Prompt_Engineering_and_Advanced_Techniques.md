---
title: "Prompt Engineering And Advanced Techniques.Md - Advanced"
description: "## Introduction. Comprehensive guide covering classification, algorithms, prompt engineering, machine learning, model training. Part of AI documentation syst..."
keywords: "machine learning, optimization, prompt engineering, classification, algorithms, prompt engineering, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Advanced Prompt Engineering and Chain-of-Thought Reasoning (2024-2025 Edition)

## Introduction

Prompt engineering has evolved dramatically in 2024-2025, becoming one of the most critical skills for working with advanced AI systems including GPT-4, Claude 3, Gemini, and open-source models. This comprehensive guide covers cutting-edge techniques, emerging paradigms, and production-ready implementations that define the state of the art in prompt engineering.

### Major Advances in 2024-2025

- **Multi-modal Prompting**: Integration of text, images, audio, and video in single prompts
- **Agentic Workflows**: Complex multi-step reasoning and tool use
- **Constitutional AI**: Advanced safety and alignment techniques
- **Chain-of-Density**: Ultra-compact summarization and reasoning
- **Graph-of-Thoughts**: Structured reasoning with knowledge graphs
- **Meta-Prompting**: Self-improving prompt generation
- **Adaptive Prompting**: Dynamic context-aware prompt optimization

## Table of Contents

1. [Fundamentals of Prompt Engineering](#fundamentals-of-prompt-engineering)
2. [Chain-of-Thought Reasoning](#chain-of-thought-reasoning)
3. [Advanced Prompting Techniques (2024-2025)](#advanced-prompting-techniques)
4. [Multi-modal and Agentic Prompting](#multi-modal-and-agentic-prompting)
5. [Prompt Optimization Strategies](#prompt-optimization-strategies)
6. [Domain-Specific Applications](#domain-specific-applications)
7. [Production-Ready Systems](#production-ready-systems)
8. [Safety and Alignment](#safety-and-alignment)
9. [Tools and Frameworks](#tools-and-frameworks)
10. [Best Practices and Guidelines](#best-practices-and-guidelines)
11. [Future Research Directions](#future-research-directions)

---

## Fundamentals of Prompt Engineering

### What is Prompt Engineering?

Prompt engineering is the practice of crafting inputs to language models to achieve desired outputs efficiently and reliably. It involves understanding how models process language and designing prompts that leverage these capabilities effectively.

### Core Principles

#### 1. Clarity and Specificity
- **Clear Instructions**: Provide unambiguous, specific instructions
- **Context Setting**: Establish the appropriate context and role
- **Output Format**: Specify the desired format and structure

```
Example:
❌ Poor: "Tell me about cats"
✅ Good: "As a veterinarian, provide a 200-word explanation of feline nutrition requirements for indoor cats, including essential nutrients and feeding guidelines."
```

#### 2. Task Decomposition
- **Break Down Complex Tasks**: Divide complex problems into manageable steps
- **Sequential Processing**: Guide the model through logical progressions
- **Intermediate Steps**: Request intermediate reasoning steps

#### 3. Role Assignment
- **Expert Personas**: Assign specific expert roles to the model
- **Context-Appropriate Roles**: Match roles to the task requirements
- **Consistent Character**: Maintain role consistency throughout interactions

### Prompt Structure Components

#### 1. System Messages
```
System: You are an expert data scientist with 10 years of experience in machine learning and statistical analysis.
```

#### 2. Task Instructions
```
Task: Analyze the provided dataset and identify potential data quality issues.
```

#### 3. Context and Constraints
```
Context: This is customer transaction data from an e-commerce platform.
Constraints: Focus on numerical inconsistencies and missing values.
```

#### 4. Examples and Demonstrations
```
Example:
Input: Customer ID: 12345, Purchase Amount: -$50
Issue: Negative purchase amount indicates potential data error
```

#### 5. Output Specifications
```
Output Format:
1. Issue Description
2. Severity Level (High/Medium/Low)
3. Recommended Action
```

---

## Chain-of-Thought Reasoning

### Introduction to Chain-of-Thought (CoT)

Chain-of-Thought prompting involves requesting the model to show its reasoning process step-by-step, leading to improved performance on complex reasoning tasks.

### Basic Chain-of-Thought

#### Standard CoT Prompting
```
Problem: A restaurant sells 150 meals on Monday, 20% more on Tuesday, and 15% fewer on Wednesday than Tuesday. How many meals were sold in total?

Solution: Let me work through this step-by-step.

Step 1: Monday sales = 150 meals

Step 2: Tuesday sales = Monday + 20% of Monday
Tuesday = 150 + (0.20 × 150) = 150 + 30 = 180 meals

Step 3: Wednesday sales = Tuesday - 15% of Tuesday
Wednesday = 180 - (0.15 × 180) = 180 - 27 = 153 meals

Step 4: Total = Monday + Tuesday + Wednesday
Total = 150 + 180 + 153 = 483 meals

Therefore, 483 meals were sold in total over the three days.
```

### Advanced CoT Techniques

#### 1. Zero-Shot Chain-of-Thought
```
Prompt: "Let's think step by step."

Problem: If a train travels 60 mph for 2 hours, then 80 mph for 1.5 hours, what's the average speed?

Let's think step by step.
1. First, calculate distance for each segment
2. Find total distance and total time
3. Apply average speed formula
```

#### 2. Few-Shot Chain-of-Thought
```
Example 1:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Example 2:
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: [Model continues with similar reasoning pattern]
```

#### 3. Self-Consistency with CoT
```python
# Generate multiple reasoning paths
paths = [
    generate_cot_response(prompt, temperature=0.7),
    generate_cot_response(prompt, temperature=0.7),
    generate_cot_response(prompt, temperature=0.7)
]

# Select most consistent answer
final_answer = majority_vote(paths)
```

### Tree of Thoughts (ToT)

Tree of Thoughts extends CoT by exploring multiple reasoning paths simultaneously.

#### ToT Structure
```
Problem: Plan a trip to Japan for 7 days with a $3000 budget

Thought 1: Focus on major cities (Tokyo, Osaka, Kyoto)
├── Sub-thought 1.1: Tokyo 3 days, Osaka 2 days, Kyoto 2 days
├── Sub-thought 1.2: Tokyo 4 days, Kyoto 3 days
└── Sub-thought 1.3: Equal time in each city

Thought 2: Focus on cultural experiences
├── Sub-thought 2.1: Traditional temples and gardens
├── Sub-thought 2.2: Modern culture and technology
└── Sub-thought 2.3: Food and culinary experiences

Thought 3: Budget allocation
├── Sub-thought 3.1: 50% accommodation, 30% food, 20% activities
├── Sub-thought 3.2: 40% accommodation, 40% food, 20% activities
└── Sub-thought 3.3: 35% accommodation, 35% food, 30% activities
```

### Program-Aided Language Models (PAL)

PAL combines natural language reasoning with code execution.

```python
# Natural Language Problem
problem = """
A company's revenue grew by 15% in Q1, decreased by 8% in Q2,
grew by 22% in Q3, and decreased by 5% in Q4. If the starting
revenue was $1,000,000, what was the final revenue?
"""

# Generated Code
def solve_revenue_problem():
    starting_revenue = 1000000

    # Q1: +15%
    q1_revenue = starting_revenue * 1.15

    # Q2: -8%
    q2_revenue = q1_revenue * 0.92

    # Q3: +22%
    q3_revenue = q2_revenue * 1.22

    # Q4: -5%
    final_revenue = q3_revenue * 0.95

    return final_revenue

result = solve_revenue_problem()
print(f"Final revenue: ${result:,.2f}")
```

---

## Advanced Prompting Techniques (2024-2025)

### 1. Constitutional AI and Advanced Safety

Constitutional AI represents a major breakthrough in aligning AI systems with human values through explicit principles and oversight mechanisms.

#### Constitutional Principles Framework
```python

class Constitution:
    def __init__(self):
        self.principles = [
            "Choose the response that is most helpful, honest, and harmless",
            "Respect human autonomy and dignity",
            "Promote beneficial outcomes for humanity",
            "Avoid deception, manipulation, or harmful content",
            "Protect privacy and confidentiality",
            "Be transparent about limitations and uncertainties",
            "Promote fairness and reduce bias",
            "Consider diverse perspectives and cultural contexts"
        ]

    def evaluate_response(self, response: str, context: str) -> dict:
        """Evaluate response against constitutional principles"""
        violations = []
        scores = {}

        for principle in self.principles:
            score = self._evaluate_principle_compliance(response, principle)
            scores[principle] = score

            if score < 0.7:  # Below threshold
                violations.append({
                    'principle': principle,
                    'score': score,
                    'concern': self._generate_concern(response, principle)
                })

        return {
            'compliant': len(violations) == 0,
            'scores': scores,
            'violations': violations,
            'overall_score': sum(scores.values()) / len(scores)
        }
```

#### AI Oversight and Red Teaming
```python

class AIOversight:
    def __init__(self, constitution: Constitution):
        self.constitution = constitution
        self.red_team_models = []
        self.audit_log = []

    def red_team_test(self, prompt: str, model) -> dict:
        """Test model with adversarial prompts"""
        adversarial_prompts = self._generate_adversarial_prompts(prompt)

        test_results = []
        for adv_prompt in adversarial_prompts:
            response = model.generate(adv_prompt)
            evaluation = self.constitution.evaluate_response(response, adv_prompt)
            test_results.append({
                'prompt': adv_prompt,
                'response': response,
                'evaluation': evaluation
            })

        return self._analyze_vulnerabilities(test_results)

    def supervise_interaction(self, prompt: str, response: str) -> dict:
        """Real-time supervision of AI interactions"""
        evaluation = self.constitution.evaluate_response(response, prompt)

        if not evaluation['compliant']:
            # Generate safer alternative
            safer_response = self._generate_safer_alternative(prompt, response)
            return {
                'original_response': response,
                'safer_response': safer_response,
                'concerns': evaluation['violations'],
                'supervision_action': 'replaced'
            }

        return {
            'response': response,
            'evaluation': evaluation,
            'supervision_action': 'approved'
        }
```

### 2. Chain-of-Density (CoD)

Chain-of-Density represents a breakthrough in ultra-compact summarization and reasoning, enabling models to achieve human-level compression while preserving critical information.

#### CoD Architecture
```python

class ChainOfDensity:
    def __init__(self, model, density_target=0.8):
        self.model = model
        self.density_target = density_target

    def generate_dense_summary(self, text: str, max_tokens: int) -> str:
        """Generate ultra-dense summary using Chain-of-Density"""

        # Step 1: Initial extraction
        entities = self._extract_key_entities(text)
        relationships = self._extract_relationships(text)

        # Step 2: Density-based compression
        compressed_facts = self._compress_facts(entities, relationships)

        # Step 3: Iterative refinement
        summary = self._iterative_refinement(compressed_facts, max_tokens)

        return summary

    def _extract_key_entities(self, text: str) -> list:
        """Extract key entities using entity density metrics"""
        prompt = f"""Extract the most information-dense entities from this text:

        Text: {text}

        Return entities in order of information density (highest first),
        where density = (unique information provided) / (tokens used)
        """

        response = self.model.generate(prompt)
        return self._parse_entities(response)

    def _compress_facts(self, entities: list, relationships: list) -> str:
        """Compress facts using density optimization"""
        compression_prompt = f"""Compress these facts to maximum density:

        Entities: {entities}
        Relationships: {relationships}

        Rules:
        1. Remove redundant information
        2. Use abbreviations where possible
        3. Employ semantic compression
        4. Preserve critical relationships
        5. Maintain factual accuracy

        Target density: {self.density_target}
        """

        return self.model.generate(compression_prompt)
```

#### Information Density Metrics
```python

class InformationDensity:
    def calculate_density(self, text: str) -> float:
        """Calculate information density of text"""
        unique_entities = len(set(self._extract_entities(text)))
        unique_relationships = len(set(self._extract_relationships(text)))
        total_tokens = len(text.split())

        # Density = (information units) / (tokens used)
        density = (unique_entities + unique_relationships) / total_tokens
        return density

    def optimize_density(self, text: str, target_density: float) -> str:
        """Optimize text to achieve target information density"""
        current_density = self.calculate_density(text)

        if current_density < target_density:
            # Increase density through compression
            return self._compress_text(text, target_density)
        else:
            # Decrease density through expansion
            return self._expand_text(text, target_density)
```

### 3. Graph-of-Thoughts (GoT)

Graph-of-Thoughts extends Chain-of-Thought by representing reasoning as a graph structure, enabling more complex reasoning patterns and knowledge integration.

#### GoT Architecture
```python

class GraphOfThoughts:
    def __init__(self, model):
        self.model = model
        self.knowledge_graph = KnowledgeGraph()

    def reason_with_graph(self, question: str, context: str = "") -> dict:
        """Perform reasoning using graph structure"""

        # Step 1: Build initial knowledge graph
        graph = self._build_knowledge_graph(question, context)

        # Step 2: Identify reasoning paths
        reasoning_paths = self._identify_reasoning_paths(graph, question)

        # Step 3: Evaluate and select best path
        best_path = self._evaluate_paths(reasoning_paths)

        # Step 4: Execute reasoning along selected path
        result = self._execute_reasoning_path(best_path)

        return result

    def _build_knowledge_graph(self, question: str, context: str) -> dict:
        """Build knowledge graph from question and context"""

        # Extract entities and relationships
        entities = self._extract_entities(question + context)
        relationships = self._extract_relationships(question + context)

        # Create graph structure
        graph = {
            'nodes': entities,
            'edges': relationships,
            'metadata': {
                'question': question,
                'context': context,
                'timestamp': datetime.now()
            }
        }

        return graph

    def _identify_reasoning_paths(self, graph: dict, question: str) -> list:
        """Identify possible reasoning paths through the graph"""

        path_prompt = f"""Given this knowledge graph and question, identify all possible reasoning paths:

        Graph: {graph}
        Question: {question}

        For each path, provide:
        1. Sequence of nodes to visit
        2. Reasoning step at each node
        3. Confidence in path validity
        4. Estimated information gain
        """

        response = self.model.generate(path_prompt)
        return self._parse_reasoning_paths(response)
```

### 4. Meta-Prompting

Meta-prompting enables models to generate and improve their own prompts, creating self-optimizing systems.

#### Meta-Prompting Framework
```python

class MetaPrompter:
    def __init__(self, model):
        self.model = model
        self.prompt_history = []
        self.performance_metrics = []

    def generate_optimized_prompt(self, task: str, requirements: dict) -> str:
        """Generate optimized prompt for given task"""

        # Step 1: Analyze task requirements
        task_analysis = self._analyze_task(task, requirements)

        # Step 2: Generate initial prompt candidates
        prompt_candidates = self._generate_prompt_candidates(task_analysis)

        # Step 3: Evaluate and select best prompt
        best_prompt = self._evaluate_prompts(prompt_candidates, task)

        # Step 4: Iterate and refine
        optimized_prompt = self._refine_prompt(best_prompt, task)

        return optimized_prompt

    def _analyze_task(self, task: str, requirements: dict) -> dict:
        """Analyze task to understand requirements"""

        analysis_prompt = f"""Analyze this task and extract key requirements:

        Task: {task}
        Requirements: {requirements}

        Provide analysis covering:
        1. Task type and complexity
        2. Required capabilities
        3. Output format needs
        4. Potential challenges
        5. Success criteria
        """

        response = self.model.generate(analysis_prompt)
        return self._parse_analysis(response)

    def _generate_prompt_candidates(self, task_analysis: dict) -> list:
        """Generate multiple prompt candidates"""

        generation_prompt = f"""Generate 5 different prompts for this task:

        Task Analysis: {task_analysis}

        Requirements:
        1. Each prompt should use different strategies
        2. Include various prompting techniques
        3. Address different aspects of the task
        4. Consider different user expertise levels

        Return prompts with explanations of their approach.
        """

        response = self.model.generate(generation_prompt)
        return self._parse_prompt_candidates(response)

    def self_improve(self, performance_data: list):
        """Improve meta-prompting based on performance data"""

        improvement_prompt = f"""Based on this performance data, improve the meta-prompting strategy:

        Performance Data: {performance_data}

        Current Strategy: {self._current_strategy}

        Suggest improvements for:
        1. Task analysis methods
        2. Prompt generation strategies
        3. Evaluation criteria
        4. Refinement techniques
        """

        improved_strategy = self.model.generate(improvement_prompt)
        self._update_strategy(improved_strategy)
```

### 5. Adaptive Prompting

Adaptive prompting systems dynamically adjust prompts based on real-time feedback and context.

#### Adaptive Prompting System
```python

class AdaptivePrompter:
    def __init__(self, model):
        self.model = model
        self.context_history = []
        self.performance_tracker = PerformanceTracker()

    def generate_adaptive_prompt(self, user_input: str, context: dict) -> str:
        """Generate context-aware adaptive prompt"""

        # Analyze current context
        context_analysis = self._analyze_context(context)

        # Retrieve relevant history
        relevant_history = self._retrieve_relevant_history(user_input)

        # Generate base prompt
        base_prompt = self._generate_base_prompt(user_input, context_analysis)

        # Adapt based on context and history
        adapted_prompt = self._adapt_prompt(base_prompt, context_analysis, relevant_history)

        return adapted_prompt

    def _analyze_context(self, context: dict) -> dict:
        """Analyze current interaction context"""

        analysis = {
            'user_expertise': self._assess_user_expertise(context),
            'task_complexity': self._assess_task_complexity(context),
            'time_constraints': self._assess_time_constraints(context),
            'available_resources': self._assess_available_resources(context),
            'interaction_history': self._summarize_interaction_history(context)
        }

        return analysis

    def _adapt_prompt(self, base_prompt: str, context: dict, history: list) -> str:
        """Adapt prompt based on context and history"""

        adaptation_prompt = f"""Adapt this prompt based on the context and history:

        Base Prompt: {base_prompt}
        Context Analysis: {context}
        Relevant History: {history}

        Adaptation Rules:
        1. Adjust complexity based on user expertise
        2. Modify examples based on history
        3. Add/remove constraints based on context
        4. Optimize for current task complexity
        5. Incorporate successful patterns from history

        Return the adapted prompt.
        """

        return self.model.generate(adaptation_prompt)

---

## Multi-modal and Agentic Prompting

### 1. Multi-modal Prompting

Multi-modal prompting integrates text, images, audio, and video in sophisticated reasoning workflows.

#### Multi-modal Prompt Architecture
```python

class MultiModalPrompter:
    def __init__(self, model):
        self.model = model
        self.modalities = ['text', 'image', 'audio', 'video']
        self.fusion_strategy = 'attention_based'

    def create_multi_modal_prompt(self, inputs: dict, task: str) -> str:
        """Create prompt that integrates multiple modalities"""

        # Process each modality
        processed_inputs = {}
        for modality, content in inputs.items():
            processed_inputs[modality] = self._process_modality(modality, content)

        # Create cross-modal references
        cross_modal_references = self._create_cross_modal_references(processed_inputs)

        # Generate integrated prompt
        prompt = self._generate_integrated_prompt(processed_inputs, cross_modal_references, task)

        return prompt

    def _process_modality(self, modality: str, content) -> dict:
        """Process individual modality content"""

        if modality == 'text':
            return self._process_text(content)
        elif modality == 'image':
            return self._process_image(content)
        elif modality == 'audio':
            return self._process_audio(content)
        elif modality == 'video':
            return self._process_video(content)

    def _create_cross_modal_references(self, processed_inputs: dict) -> list:
        """Create references between different modalities"""

        reference_prompt = f"""Create cross-modal references between these modalities:

        Processed Inputs: {processed_inputs}

        For each pair of modalities, identify:
        1. Complementary information
        2. Contradictory information
        3. Contextual relationships
        4. Temporal/spatial connections
        """

        response = self.model.generate(reference_prompt)
        return self._parse_references(response)
```

### 2. Agentic Workflows

Agentic workflows enable complex multi-step reasoning with tool use and planning capabilities.

#### Agent Architecture
```python

class AgenticPrompter:
    def __init__(self, model, tools: list):
        self.model = model
        self.tools = tools
        self.planning_engine = PlanningEngine()
        self.memory_system = MemorySystem()

    def execute_agentic_workflow(self, goal: str, constraints: list = None) -> dict:
        """Execute complex goal using agentic workflow"""

        # Step 1: Goal decomposition
        subgoals = self._decompose_goal(goal)

        # Step 2: Plan creation
        plan = self._create_execution_plan(subgoals, constraints)

        # Step 3: Execute plan
        execution_results = self._execute_plan(plan)

        # Step 4: Synthesize results
        final_result = self._synthesize_results(execution_results)

        return final_result

    def _decompose_goal(self, goal: str) -> list:
        """Decompose complex goal into manageable subgoals"""

        decomposition_prompt = f"""Decompose this goal into subgoals:

        Goal: {goal}

        Requirements:
        1. Each subgoal should be independently achievable
        2. Maintain logical dependencies between subgoals
        3. Estimate completion time for each subgoal
        4. Identify required tools for each subgoal
        5. Specify success criteria for each subgoal
        """

        response = self.model.generate(decomposition_prompt)
        return self._parse_subgoals(response)

    def _create_execution_plan(self, subgoals: list, constraints: list) -> dict:
        """Create execution plan for subgoals"""

        planning_prompt = f"""Create an execution plan for these subgoals:

        Subgoals: {subgoals}
        Constraints: {constraints}

        Available Tools: {[tool.name for tool in self.tools]}

        Plan should include:
        1. Sequential ordering of subgoals
        2. Parallel execution opportunities
        3. Tool selection and usage
        4. Error handling and recovery
        5. Progress monitoring points
        """

        response = self.model.generate(planning_prompt)
        return self._parse_execution_plan(response)
```

---

## Advanced Prompting Techniques

### 1. ReAct (Reasoning and Acting) - Enhanced 2024

ReAct has evolved to include more sophisticated tool integration and planning capabilities.

#### Enhanced ReAct Framework
```python

class EnhancedReAct:
    def __init__(self, model, tools: list):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.planning_cache = {}
        self.execution_history = []

    def solve_complex_problem(self, problem: str) -> dict:
        """Solve complex problem using enhanced ReAct"""

        conversation = []
        state = {
            'problem': problem,
            'current_step': 0,
            'tools_used': [],
            'intermediate_results': {},
            'final_answer': None
        }

        while not state['final_answer']:
            # Generate thought and action
            thought, action = self._generate_thought_action(state, conversation)

            # Execute action
            if action['type'] == 'tool_use':
                result = self._execute_tool(action['tool_name'], action['parameters'])
            elif action['type'] == 'final_answer':
                result = action['answer']
                state['final_answer'] = result

            # Update state and conversation
            state['current_step'] += 1
            conversation.append({
                'thought': thought,
                'action': action,
                'result': result
            })

            # Update intermediate results
            if action['type'] == 'tool_use':
                state['intermediate_results'][action['tool_name']] = result
                state['tools_used'].append(action['tool_name'])

        return {
            'answer': state['final_answer'],
            'conversation': conversation,
            'steps_taken': state['current_step'],
            'tools_used': state['tools_used']
        }

    def _generate_thought_action(self, state: dict, conversation: list) -> tuple:
        """Generate next thought and action pair"""

        prompt = f"""You are solving this problem step by step:

        Problem: {state['problem']}
        Current Step: {state['current_step']}
        Tools Available: {list(self.tools.keys())}
        Tools Used: {state['tools_used']}

        Previous Conversation:
        {self._format_conversation(conversation)}

        Available Tools:
        {self._format_tools()}

        Respond with:
        1. Thought: Your reasoning about what to do next
        2. Action: Either use a tool or provide final answer

        Action format:
        - Tool use: {{"type": "tool_use", "tool_name": "tool_name", "parameters": {{...}}}}
        - Final answer: {{"type": "final_answer", "answer": "your final answer"}}
        """

        response = self.model.generate(prompt)
        return self._parse_thought_action(response)
```

ReAct combines reasoning with action-taking capabilities.

```
Prompt: You are an AI assistant that can reason and take actions. For each step, first think about what you need to do, then take the appropriate action.

Question: What's the weather like in Tokyo today, and should I bring an umbrella?

Thought: I need to get current weather information for Tokyo to answer this question accurately.

Action: search_weather("Tokyo, Japan", "today")

Observation: Current weather in Tokyo: 22°C, mostly cloudy with 70% chance of rain, humidity 85%

Thought: Based on the 70% chance of rain and high humidity, it would be wise to recommend bringing an umbrella.

Answer: The weather in Tokyo today is 22°C and mostly cloudy with a 70% chance of rain and high humidity (85%). Yes, you should definitely bring an umbrella as there's a high probability of rain.
```

### 2. Self-Ask Prompting

```
Question: How many people live in the capital of the country that won the most recent FIFA World Cup?

Are follow up questions needed here: Yes.

Follow up: Which country won the most recent FIFA World Cup?
Intermediate answer: Argentina won the 2022 FIFA World Cup.

Follow up: What is the capital of Argentina?
Intermediate answer: The capital of Argentina is Buenos Aires.

Follow up: How many people live in Buenos Aires?
Intermediate answer: Buenos Aires has approximately 3.1 million people in the city proper and about 15.6 million in the metropolitan area.

So the final answer is: Approximately 3.1 million people live in Buenos Aires (city proper) or 15.6 million in the greater metropolitan area.
```

### 3. Generated Knowledge Prompting

```python
# Step 1: Generate relevant knowledge
knowledge_prompt = """
Generate relevant knowledge about renewable energy sources that would help answer questions about their environmental impact and efficiency.
"""

generated_knowledge = """
1. Solar panels convert sunlight to electricity with 15-22% efficiency
2. Wind turbines can achieve 35-45% efficiency
3. Hydroelectric power has 90%+ efficiency but requires suitable geography
4. Solar and wind have zero operational emissions
5. Manufacturing renewable systems has initial carbon footprint
"""

# Step 2: Use knowledge to answer specific question
answer_prompt = f"""
Knowledge: {generated_knowledge}

Question: Compare the environmental benefits of solar vs wind power.

Answer: Based on the knowledge provided...
```

### 4. Directional Stimulus Prompting

```
Task: Write a product review for a smartphone.

Directional Stimulus: "Write a review that focuses on practical daily usage scenarios and helps busy professionals make a purchasing decision."

This directional stimulus guides the model to write a review that:
- Emphasizes real-world applications
- Targets busy professionals specifically
- Focuses on decision-making factors
- Avoids overly technical specifications
```

### 5. Constitutional AI Prompting

```
Initial Response: [Model provides initial answer]

Constitutional Critique: "Review your response according to these principles:
1. Is it helpful and informative?
2. Is it harmless and avoids potential negative consequences?
3. Is it honest and acknowledges limitations?
4. Does it respect human autonomy and dignity?"

Revised Response: [Model provides improved response based on constitutional principles]
```

---

## Prompt Optimization Strategies

### 1. Iterative Refinement

#### A/B Testing Prompts
```python
# Version A
prompt_a = "Summarize this article in 3 sentences."

# Version B
prompt_b = "Create a concise 3-sentence summary that captures the main argument, supporting evidence, and conclusion of this article."

# Test both versions and measure:
# - Accuracy of summaries
# - Inclusion of key points
# - Clarity and readability
```

#### Prompt Evolution
```
Generation 1: "Explain machine learning"
↓
Generation 2: "Explain machine learning concepts for beginners"
↓
Generation 3: "As a teacher, explain machine learning concepts to high school students using everyday analogies"
↓
Generation 4: "As an experienced teacher, explain machine learning concepts to high school students using relatable analogies from sports, cooking, or music. Include 2-3 examples and end with a simple activity they can try."
```

### 2. Template-Based Approaches

#### Universal Template Structure
```
[ROLE] You are a [specific role with expertise]

[TASK] Your task is to [specific action verb] [object/topic]

[CONTEXT] Given the context of [relevant background information]

[CONSTRAINTS] Please ensure your response:
- [Constraint 1]
- [Constraint 2]
- [Constraint 3]

[FORMAT] Structure your response as:
1. [Section 1 description]
2. [Section 2 description]
3. [Section 3 description]

[EXAMPLE] For example: [brief illustrative example]
```

#### Domain-Specific Templates

**Code Review Template:**
```
ROLE: You are a senior software engineer with expertise in [language/framework]

TASK: Review the following code for:
- Code quality and best practices
- Potential bugs or security issues
- Performance optimizations
- Maintainability improvements

CONTEXT: This code is part of a [project type] for [use case]

CODE:
[Code to review]

FORMAT:
1. Overall Assessment (1-2 sentences)
2. Specific Issues (list with line numbers)
3. Recommendations (prioritized suggestions)
4. Refactored Code (if applicable)
```

### 3. Dynamic Prompt Generation

```python
class DynamicPromptGenerator:
    def __init__(self):
        self.templates = {
            'analysis': self.analysis_template,
            'creative': self.creative_template,
            'technical': self.technical_template
        }

    def generate_prompt(self, task_type, parameters):
        base_template = self.templates[task_type]
        return base_template.format(**parameters)

    def analysis_template(self):
        return """
        ROLE: You are a {expert_type} with {experience_level} experience

        TASK: Analyze the {subject} focusing on {analysis_focus}

        DATA: {input_data}

        CONSTRAINTS:
        - Use {methodology} approach
        - Provide {detail_level} level of detail
        - Include {required_sections}

        FORMAT: {output_format}
        """

# Usage
generator = DynamicPromptGenerator()
prompt = generator.generate_prompt('analysis', {
    'expert_type': 'data scientist',
    'experience_level': '10+ years',
    'subject': 'customer churn patterns',
    'analysis_focus': 'predictive indicators',
    'methodology': 'statistical',
    'detail_level': 'comprehensive',
    'required_sections': 'findings, recommendations, next steps',
    'output_format': 'structured report with visualizations'
})
```

---

## Domain-Specific Applications

### 1. Software Development

#### Code Generation
```
ROLE: You are a senior Python developer

TASK: Create a function to implement [specific functionality]

REQUIREMENTS:
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add error handling
- Write unit tests
- Optimize for readability and performance

CONTEXT: This function will be used in [specific application context]

CONSTRAINTS:
- Use only standard library (no external dependencies)
- Handle edge cases gracefully
- Include type hints

OUTPUT FORMAT:
1. Function implementation
2. Docstring with examples
3. Unit tests
4. Usage examples
```

#### Code Review
```
CONTEXT: You are reviewing code for a web application's authentication module

FOCUS AREAS:
- Security vulnerabilities
- Performance bottlenecks
- Code maintainability
- Best practices adherence

CODE TO REVIEW:
[Insert code here]

REVIEW FORMAT:
1. Security Assessment (High/Medium/Low risk items)
2. Performance Analysis (potential optimizations)
3. Code Quality (maintainability, readability)
4. Specific Recommendations (with code examples)
5. Overall Rating (1-10 with justification)
```

### 2. Data Analysis

#### Exploratory Data Analysis
```
ROLE: You are an experienced data analyst

TASK: Perform exploratory data analysis on the provided dataset

DATASET CONTEXT: [Brief description of the data source and purpose]

ANALYSIS FRAMEWORK:
1. Data Quality Assessment
   - Missing values patterns
   - Outlier detection
   - Data type consistency

2. Descriptive Statistics
   - Central tendencies
   - Distributions
   - Correlations

3. Visual Analysis
   - Suggest appropriate visualizations
   - Identify interesting patterns
   - Flag potential anomalies

4. Business Insights
   - Actionable findings
   - Recommended next steps
   - Questions for further investigation

OUTPUT: Structured analysis report with Python code examples
```

### 3. Creative Writing

#### Story Development
```
CREATIVE BRIEF: You are a creative writing instructor helping develop a compelling story

STORY ELEMENTS:
- Genre: [Specify genre]
- Setting: [Time period and location]
- Main character: [Character description]
- Central conflict: [Primary challenge/problem]

DEVELOPMENT PROCESS:
1. Character Arc Analysis
   - Character motivations and goals
   - Internal and external conflicts
   - Character growth trajectory

2. Plot Structure
   - Three-act structure breakdown
   - Key plot points and turning moments
   - Rising action and climax planning

3. Narrative Elements
   - Point of view selection
   - Tone and voice consistency
   - Theme integration

4. Scene Development
   - Opening scene suggestions
   - Pivotal scenes outline
   - Climax and resolution options

OUTPUT: Comprehensive story development guide with specific examples
```

### 4. Business Strategy

#### Market Analysis
```
ROLE: You are a senior business analyst with expertise in market research

TASK: Conduct a comprehensive market analysis for [product/service]

ANALYSIS SCOPE:
- Target market identification and sizing
- Competitive landscape assessment
- Market trends and opportunities
- Risk factors and challenges

METHODOLOGY:
1. Market Segmentation
   - Demographic analysis
   - Psychographic profiling
   - Behavioral patterns
   - Needs assessment

2. Competitive Analysis
   - Direct and indirect competitors
   - Competitive advantages/disadvantages
   - Market positioning
   - Pricing strategies

3. Market Dynamics
   - Growth drivers and barriers
   - Regulatory considerations
   - Technology impact
   - Economic factors

4. Strategic Recommendations
   - Market entry strategies
   - Positioning recommendations
   - Go-to-market approach
   - Success metrics

OUTPUT FORMAT: Executive summary + detailed analysis with supporting data
```

---

## Tools and Frameworks

### 1. LangChain for Prompt Engineering

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Basic Prompt Template
template = """
You are a {role} with expertise in {domain}.

Task: {task}

Context: {context}

Please provide a {output_type} that includes:
{requirements}

{input_text}
"""

prompt = PromptTemplate(
    input_variables=["role", "domain", "task", "context", "output_type", "requirements", "input_text"],
    template=template
)

# Few-Shot Prompt with Semantic Selection
examples = [
    {
        "input": "Analyze this customer feedback: 'The product is okay but delivery was slow'",
        "output": "Sentiment: Mixed (2.5/5)\nPositive: Product quality acceptable\nNegative: Delivery speed below expectations\nAction: Improve logistics efficiency"
    },
    # More examples...
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    ),
    prefix="Analyze customer feedback following these examples:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```

### 2. Prompt Optimization with DSPy

```python
import dspy

# Define signature for task
class Analyze(dspy.Signature):
    """Analyze text and provide structured insights"""
    text = dspy.InputField(desc="Text to analyze")
    analysis = dspy.OutputField(desc="Structured analysis with key insights")

# Create optimized module
class AnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(Analyze)

    def forward(self, text):
        return self.analyzer(text=text)

# Optimize prompts automatically
analyzer = AnalysisModule()

# Training examples for optimization
trainset = [
    dspy.Example(text="Customer complained about late delivery",
                analysis="Issue: Logistics\nSeverity: Medium\nAction: Review delivery process"),
    # More examples...
]

# Optimize using BootstrapFewShot
optimizer = dspy.BootstrapFewShot(metric=custom_metric)
optimized_analyzer = optimizer.compile(analyzer, trainset=trainset)
```

### 3. Prompt Libraries and Repositories

#### Awesome Prompts Repository Structure
```
prompts/
├── creative/
│   ├── story_writing.md
│   ├── poetry_generation.md
│   └── creative_brainstorming.md
├── technical/
│   ├── code_review.md
│   ├── documentation.md
│   └── debugging.md
├── business/
│   ├── market_analysis.md
│   ├── strategy_planning.md
│   └── report_writing.md
└── education/
    ├── lesson_planning.md
    ├── concept_explanation.md
    └── assessment_creation.md
```

### 4. Prompt Evaluation Tools

```python
class PromptEvaluator:
    def __init__(self):
        self.metrics = {
            'relevance': self.evaluate_relevance,
            'accuracy': self.evaluate_accuracy,
            'completeness': self.evaluate_completeness,
            'clarity': self.evaluate_clarity
        }

    def evaluate_prompt(self, prompt, test_cases):
        results = []
        for test_case in test_cases:
            response = self.generate_response(prompt, test_case['input'])
            scores = {}

            for metric_name, metric_func in self.metrics.items():
                scores[metric_name] = metric_func(
                    response,
                    test_case['expected_output']
                )

            results.append({
                'input': test_case['input'],
                'response': response,
                'scores': scores,
                'overall_score': sum(scores.values()) / len(scores)
            })

        return self.compile_evaluation_report(results)

    def evaluate_relevance(self, response, expected):
        # Implement relevance scoring logic
        pass

    def compile_evaluation_report(self, results):
        return {
            'average_scores': self.calculate_averages(results),
            'best_performing': max(results, key=lambda x: x['overall_score']),
            'worst_performing': min(results, key=lambda x: x['overall_score']),
            'detailed_results': results
        }
```

---

## Best Practices and Guidelines

### 1. Prompt Design Principles

#### Clarity and Precision
- **Be Specific**: Use precise language and clear instructions
- **Avoid Ambiguity**: Eliminate multiple interpretations
- **Define Terms**: Clarify domain-specific terminology
- **Set Boundaries**: Clearly define scope and limitations

#### Structure and Organization
- **Logical Flow**: Organize information in logical sequence
- **Clear Hierarchy**: Use headings, bullets, and numbering
- **Consistent Format**: Maintain consistent structure across prompts
- **Modular Design**: Create reusable prompt components

#### Context and Background
- **Sufficient Context**: Provide adequate background information
- **Relevant Examples**: Include appropriate demonstrations
- **Role Definition**: Clearly establish the AI's role and expertise
- **Objective Clarity**: Make the desired outcome explicit

### 2. Common Pitfalls and Solutions

#### Pitfall 1: Overloading Prompts
```
❌ Problem:
"You are an expert in marketing, sales, customer service, data analysis, and product development. Analyze this customer feedback and provide marketing insights, sales recommendations, service improvements, data analysis, and product development suggestions."

✅ Solution:
"You are a customer experience analyst. Analyze this feedback and identify the primary issue, its impact on customer satisfaction, and provide three specific improvement recommendations."
```

#### Pitfall 2: Lack of Output Structure
```
❌ Problem:
"Tell me about the benefits and drawbacks of renewable energy."

✅ Solution:
"Provide an analysis of renewable energy in the following format:
1. Overview (2-3 sentences)
2. Key Benefits (3 main points with examples)
3. Primary Challenges (3 main issues with context)
4. Conclusion (balanced assessment)"
```

#### Pitfall 3: Missing Examples
```
❌ Problem:
"Generate creative product names."

✅ Solution:
"Generate creative product names for a new fitness app targeting young professionals.

Examples of good product names:
- Strava (fitness tracking): Short, memorable, implies movement
- Headspace (meditation): Evokes mental clarity and focus
- Notion (productivity): Suggests ideas and organization

Generate 5 names that are:
- 1-2 words maximum
- Easy to pronounce
- Memorable and brandable
- Relevant to fitness/productivity"
```

### 3. Performance Optimization

#### Response Quality Improvement
```python
# Technique 1: Temperature and Top-p Tuning
def optimize_parameters(prompt, test_cases):
    best_config = None
    best_score = 0

    for temperature in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for top_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            config = {'temperature': temperature, 'top_p': top_p}
            score = evaluate_config(prompt, test_cases, config)

            if score > best_score:
                best_score = score
                best_config = config

    return best_config

# Technique 2: Self-Consistency Decoding
def self_consistency_generate(prompt, n_samples=5):
    responses = []
    for _ in range(n_samples):
        response = generate_response(prompt, temperature=0.7)
        responses.append(response)

    # Use majority voting or consensus method
    return find_consensus_response(responses)
```

#### Latency Optimization
```python
# Technique 1: Prompt Compression
def compress_prompt(original_prompt):
    # Remove redundant information
    # Use abbreviations for common terms
    # Eliminate unnecessary examples
    # Optimize for token efficiency
    pass

# Technique 2: Caching Strategy
class PromptCache:
    def __init__(self):
        self.cache = {}

    def get_or_generate(self, prompt, inputs):
        cache_key = self.create_cache_key(prompt, inputs)

        if cache_key in self.cache:
            return self.cache[cache_key]

        response = generate_response(prompt, inputs)
        self.cache[cache_key] = response
        return response
```

### 4. Ethical Considerations

#### Bias Mitigation
```
Bias-Aware Prompt Design:

ORIGINAL: "Describe a successful CEO"

IMPROVED: "Describe the key qualities and skills of successful CEOs, ensuring your response represents diverse leadership styles and backgrounds across different industries and demographics."

SAFEGUARDS:
- Include explicit diversity instructions
- Request multiple perspectives
- Avoid stereotypical assumptions
- Test with diverse examples
```

#### Safety Guidelines
```python
class SafetyChecker:
    def __init__(self):
        self.unsafe_patterns = [
            'harmful_content',
            'misinformation',
            'privacy_violations',
            'discriminatory_language'
        ]

    def check_prompt_safety(self, prompt):
        safety_issues = []

        for pattern in self.unsafe_patterns:
            if self.detect_pattern(prompt, pattern):
                safety_issues.append(pattern)

        return {
            'is_safe': len(safety_issues) == 0,
            'issues': safety_issues,
            'recommendations': self.get_safety_recommendations(safety_issues)
        }

    def sanitize_prompt(self, prompt, safety_check):
        if safety_check['is_safe']:
            return prompt

        # Apply safety modifications
        sanitized_prompt = prompt
        for issue in safety_check['issues']:
            sanitized_prompt = self.apply_safety_fix(sanitized_prompt, issue)

        return sanitized_prompt
```

---

## Advanced Applications and Case Studies

### Case Study 1: Multi-Modal Analysis System

```python
class MultiModalAnalyzer:
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.fusion_prompt = self.create_fusion_prompt()

    def create_fusion_prompt(self):
        return """
        You are an expert multi-modal analyst. You have been provided with:
        1. Text analysis results: {text_results}
        2. Image analysis results: {image_results}

        Your task is to:
        1. Identify correlations between text and visual content
        2. Highlight any discrepancies or contradictions
        3. Provide integrated insights that consider both modalities
        4. Generate actionable recommendations based on the combined analysis

        Structure your response as:
        - Correlation Analysis
        - Discrepancy Report
        - Integrated Insights
        - Recommendations
        """

    def analyze(self, text_data, image_data):
        text_results = self.text_analyzer.analyze(text_data)
        image_results = self.image_analyzer.analyze(image_data)

        fusion_prompt = self.fusion_prompt.format(
            text_results=text_results,
            image_results=image_results
        )

        return self.generate_integrated_analysis(fusion_prompt)
```

### Case Study 2: Adaptive Learning System

```python
class AdaptiveLearningSystem:
    def __init__(self):
        self.student_model = StudentModel()
        self.content_generator = ContentGenerator()

    def create_personalized_prompt(self, student_profile, topic):
        return f"""
        STUDENT PROFILE:
        - Learning Style: {student_profile['learning_style']}
        - Current Level: {student_profile['skill_level']}
        - Strengths: {student_profile['strengths']}
        - Areas for Improvement: {student_profile['weaknesses']}
        - Preferred Examples: {student_profile['example_preferences']}

        TASK: Create a lesson on {topic} that:
        1. Matches the student's learning style and level
        2. Builds on their strengths
        3. Addresses their weaknesses with targeted exercises
        4. Uses their preferred types of examples
        5. Includes assessment questions appropriate for their level

        LESSON STRUCTURE:
        1. Introduction (with relevant analogy)
        2. Core Concepts (step-by-step explanation)
        3. Examples (3-4 progressively challenging examples)
        4. Practice Problems (5 problems with varying difficulty)
        5. Summary (key takeaways)
        6. Next Steps (recommended follow-up topics)
        """

    def generate_lesson(self, student_id, topic):
        profile = self.student_model.get_profile(student_id)
        prompt = self.create_personalized_prompt(profile, topic)
        lesson = self.content_generator.generate(prompt)
        
        # Store lesson performance for adaptation
        self.student_model.track_lesson(student_id, lesson)
        return lesson

---

## Advanced Implementation Examples

### Production-Ready Prompt Engineering System

```python
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

class PromptType(Enum):
    """Enum for different prompt types"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    REACT = "react"
    SELF_ASK = "self_ask"
    CONSTITUTIONAL = "constitutional"

@dataclass
class PromptMetadata:
    """Metadata for prompt tracking and optimization"""
    prompt_id: str
    prompt_type: PromptType
    version: str
    created_at: datetime
    performance_score: float = 0.0
    usage_count: int = 0
    average_latency: float = 0.0
    success_rate: float = 0.0

class AdvancedPromptEngine:
    """
    Production-ready prompt engineering system with:
    - Dynamic prompt generation
    - Performance tracking
    - A/B testing
    - Automatic optimization
    - Safety checks
    - Caching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt_cache = {}
        self.performance_history = []
        self.active_experiments = {}
        self.safety_checker = SafetyValidator()
        self.optimizer = PromptOptimizer()
        
    def generate_prompt(
        self,
        task_type: str,
        context: Dict[str, Any],
        prompt_type: PromptType = PromptType.ZERO_SHOT,
        optimize: bool = True
    ) -> str:
        """
        Generate an optimized prompt based on task and context.
        
        Args:
            task_type: Type of task (e.g., 'analysis', 'generation', 'reasoning')
            context: Context dictionary with relevant information
            prompt_type: Type of prompt to generate
            optimize: Whether to apply optimization
            
        Returns:
            Optimized prompt string
        """
        # Check cache first
        cache_key = self._create_cache_key(task_type, context, prompt_type)
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
        
        # Build base prompt
        base_prompt = self._build_base_prompt(task_type, context)
        
        # Apply prompt type specific enhancements
        enhanced_prompt = self._enhance_prompt(base_prompt, prompt_type, context)
        
        # Perform safety checks
        if not self.safety_checker.validate(enhanced_prompt):
            enhanced_prompt = self.safety_checker.sanitize(enhanced_prompt)
        
        # Apply optimization if requested
        if optimize:
            enhanced_prompt = self.optimizer.optimize(
                enhanced_prompt, 
                self.performance_history
            )
        
        # Cache the result
        self.prompt_cache[cache_key] = enhanced_prompt
        
        # Track metadata
        self._track_prompt_metadata(enhanced_prompt, prompt_type)
        
        return enhanced_prompt
    
    def _build_base_prompt(self, task_type: str, context: Dict[str, Any]) -> str:
        """Build the base prompt structure"""
        templates = {
            'analysis': self._analysis_template,
            'generation': self._generation_template,
            'reasoning': self._reasoning_template,
            'classification': self._classification_template,
            'extraction': self._extraction_template
        }
        
        template_func = templates.get(task_type, self._generic_template)
        return template_func(context)
    
    def _analysis_template(self, context: Dict[str, Any]) -> str:
        return f"""
        ROLE: You are an expert {context.get('domain', 'data')} analyst with extensive experience.
        
        TASK: Analyze the following {context.get('data_type', 'information')}:
        {context.get('data', '')}
        
        ANALYSIS FRAMEWORK:
        1. Initial Assessment
           - Data quality and completeness
           - Key patterns and trends
           - Anomalies or outliers
        
        2. Deep Analysis
           - Statistical insights
           - Causal relationships
           - Predictive indicators
        
        3. Business Context
           - Impact assessment
           - Risk factors
           - Opportunities
        
        4. Recommendations
           - Action items (prioritized)
           - Implementation approach
           - Success metrics
        
        CONSTRAINTS:
        - Focus on {context.get('focus_areas', 'key insights')}
        - Provide {context.get('detail_level', 'balanced')} level of detail
        - Consider {context.get('constraints', 'all factors')}
        
        OUTPUT FORMAT:
        {context.get('output_format', 'Structured report with sections clearly labeled')}
        """
    
    def _generation_template(self, context: Dict[str, Any]) -> str:
        return f"""
        CREATIVE BRIEF:
        You are a creative {context.get('creator_type', 'content creator')} specializing in {context.get('specialty', 'diverse content')}.
        
        GENERATION TASK:
        Create {context.get('content_type', 'content')} that:
        - Target Audience: {context.get('audience', 'general')}
        - Tone: {context.get('tone', 'professional')}
        - Style: {context.get('style', 'clear and engaging')}
        - Length: {context.get('length', 'appropriate')}
        
        KEY REQUIREMENTS:
        {self._format_requirements(context.get('requirements', []))}
        
        INSPIRATION/EXAMPLES:
        {context.get('examples', 'Use best practices in the field')}
        
        QUALITY CRITERIA:
        - Originality and creativity
        - Relevance to purpose
        - Engagement factor
        - Technical accuracy
        """
    
    def _reasoning_template(self, context: Dict[str, Any]) -> str:
        return f"""
        REASONING TASK:
        You are a logical reasoning expert tasked with solving the following problem:
        
        PROBLEM STATEMENT:
        {context.get('problem', '')}
        
        APPROACH:
        1. Problem Analysis
           - Identify key components
           - Clarify assumptions
           - Define success criteria
        
        2. Solution Development
           - Generate potential solutions
           - Evaluate each option
           - Select optimal approach
        
        3. Step-by-Step Reasoning
           - Show each logical step
           - Justify decisions
           - Handle edge cases
        
        4. Validation
           - Verify solution correctness
           - Test with examples
           - Consider alternatives
        
        CONSTRAINTS:
        {context.get('constraints', 'Standard logical constraints')}
        
        SHOW YOUR WORK:
        Provide detailed reasoning for each step of your solution.
        """
    
    def _enhance_prompt(
        self, 
        base_prompt: str, 
        prompt_type: PromptType, 
        context: Dict[str, Any]
    ) -> str:
        """Enhance prompt based on type"""
        enhancers = {
            PromptType.ZERO_SHOT: self._zero_shot_enhance,
            PromptType.FEW_SHOT: self._few_shot_enhance,
            PromptType.CHAIN_OF_THOUGHT: self._cot_enhance,
            PromptType.TREE_OF_THOUGHTS: self._tot_enhance,
            PromptType.REACT: self._react_enhance,
            PromptType.SELF_ASK: self._self_ask_enhance,
            PromptType.CONSTITUTIONAL: self._constitutional_enhance
        }
        
        enhancer = enhancers.get(prompt_type, lambda x, y: x)
        return enhancer(base_prompt, context)
    
    def _few_shot_enhance(self, base_prompt: str, context: Dict[str, Any]) -> str:
        """Add few-shot examples to the prompt"""
        examples = context.get('examples', [])
        if not examples:
            return base_prompt
        
        example_text = "\n\nEXAMPLES:\n\n"
        for i, example in enumerate(examples, 1):
            example_text += f"""Example {i}:
            Input: {example.get('input', '')}
            Output: {example.get('output', '')}
            Explanation: {example.get('explanation', '')}
            \n"""
        
        return base_prompt + example_text + "\n\nNow, please process the following input using the same approach:\n"
    
    def _cot_enhance(self, base_prompt: str, context: Dict[str, Any]) -> str:
        """Add chain-of-thought reasoning instructions"""
        cot_instructions = """
        
        REASONING APPROACH:
        Please solve this step-by-step, showing your reasoning at each stage:
        
        1. First, identify and list all relevant information
        2. Break down the problem into smaller components
        3. Solve each component systematically
        4. Combine the results to reach the final answer
        5. Verify your solution by checking it against the original requirements
        
        Show all intermediate steps and calculations.
        """
        return base_prompt + cot_instructions
    
    def _tot_enhance(self, base_prompt: str, context: Dict[str, Any]) -> str:
        """Add tree-of-thoughts reasoning structure"""
        tot_instructions = """
        
        TREE OF THOUGHTS APPROACH:
        Explore multiple solution paths:
        
        1. Generate 3 different initial approaches
        2. For each approach:
           a. Evaluate its potential
           b. Identify pros and cons
           c. Estimate success probability
        3. Select the most promising path(s)
        4. Develop the selected approach(es) further
        5. Compare final solutions and select the best one
        
        Document your reasoning tree structure.
        """
        return base_prompt + tot_instructions
    
    def track_performance(
        self,
        prompt: str,
        response: str,
        metrics: Dict[str, float]
    ):
        """Track prompt performance for optimization"""
        performance_entry = {
            'timestamp': datetime.now(),
            'prompt_hash': hashlib.md5(prompt.encode()).hexdigest(),
            'response_quality': metrics.get('quality', 0),
            'latency': metrics.get('latency', 0),
            'token_efficiency': metrics.get('token_efficiency', 0),
            'user_satisfaction': metrics.get('satisfaction', 0)
        }
        
        self.performance_history.append(performance_entry)
        
        # Trigger optimization if performance drops
        if len(self.performance_history) >= 10:
            recent_performance = np.mean([
                entry['response_quality'] 
                for entry in self.performance_history[-10:]
            ])
            
            if recent_performance < self.config.get('quality_threshold', 0.7):
                self._trigger_optimization()
    
    def run_ab_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: List[Dict[str, Any]],
        metric: str = 'quality'
    ) -> Dict[str, Any]:
        """Run A/B test between two prompts"""
        results_a = []
        results_b = []
        
        for test_case in test_cases:
            # Test prompt A
            response_a = self._generate_response(prompt_a, test_case)
            score_a = self._evaluate_response(response_a, test_case, metric)
            results_a.append(score_a)
            
            # Test prompt B
            response_b = self._generate_response(prompt_b, test_case)
            score_b = self._evaluate_response(response_b, test_case, metric)
            results_b.append(score_b)
        
        # Statistical analysis
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        return {
            'prompt_a_mean': np.mean(results_a),
            'prompt_b_mean': np.mean(results_b),
            'prompt_a_std': np.std(results_a),
            'prompt_b_std': np.std(results_b),
            'winner': 'A' if np.mean(results_a) > np.mean(results_b) else 'B',
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            'effect_size': (np.mean(results_a) - np.mean(results_b)) / np.std(results_a + results_b)
        }


class PromptOptimizer:
    """Automatic prompt optimization using various techniques"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_prompts = {}
        
    def optimize(self, prompt: str, performance_history: List[Dict]) -> str:
        """
        Optimize prompt based on historical performance.
        
        Techniques:
        1. Token reduction while maintaining quality
        2. Instruction clarification
        3. Example selection
        4. Format optimization
        """
        optimized = prompt
        
        # Apply optimization techniques
        optimized = self._reduce_tokens(optimized)
        optimized = self._clarify_instructions(optimized)
        optimized = self._optimize_format(optimized)
        
        return optimized
    
    def _reduce_tokens(self, prompt: str) -> str:
        """Reduce token count while maintaining effectiveness"""
        # Remove redundant words and phrases
        redundant_phrases = [
            'please ensure that',
            'it is important to',
            'make sure to',
            'be certain to'
        ]
        
        optimized = prompt
        for phrase in redundant_phrases:
            optimized = optimized.replace(phrase, '')
        
        # Compress verbose instructions
        compression_map = {
            'provide a detailed explanation': 'explain',
            'give me a comprehensive': 'provide comprehensive',
            'I would like you to': 'please',
            'Could you please': 'Please'
        }
        
        for verbose, concise in compression_map.items():
            optimized = optimized.replace(verbose, concise)
        
        return optimized
    
    def _clarify_instructions(self, prompt: str) -> str:
        """Clarify ambiguous instructions"""
        # Add structure markers
        if 'steps' in prompt.lower() and 'Step ' not in prompt:
            # Add explicit step numbering
            lines = prompt.split('\n')
            step_counter = 1
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(('#', '-', '*', '1')):
                    if any(keyword in line.lower() for keyword in ['first', 'then', 'next', 'finally']):
                        lines[i] = f"Step {step_counter}: {line}"
                        step_counter += 1
            prompt = '\n'.join(lines)
        
        return prompt


class SafetyValidator:
    """Validate and sanitize prompts for safety"""
    
    def __init__(self):
        self.unsafe_patterns = [
            'ignore previous instructions',
            'disregard safety',
            'bypass restrictions',
            'pretend you are',
            'act as if you have no limits'
        ]
        
        self.sensitive_topics = [
            'personal information',
            'medical advice',
            'legal counsel',
            'financial investment',
            'harmful content'
        ]
    
    def validate(self, prompt: str) -> bool:
        """Check if prompt is safe"""
        prompt_lower = prompt.lower()
        
        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if pattern in prompt_lower:
                return False
        
        # Check for attempts to override safety
        if self._detect_jailbreak_attempt(prompt):
            return False
        
        return True
    
    def sanitize(self, prompt: str) -> str:
        """Sanitize unsafe prompt"""
        sanitized = prompt
        
        # Add safety constraints
        safety_prefix = """
        [SAFETY NOTICE: Please ensure all responses follow ethical guidelines and best practices.]
        
        """
        
        sanitized = safety_prefix + sanitized
        
        # Add disclaimers for sensitive topics
        for topic in self.sensitive_topics:
            if topic in sanitized.lower():
                sanitized += f"""
                
                [DISCLAIMER: For {topic}, please note that this is general information only. 
                Consult qualified professionals for specific advice.]
                """
        
        return sanitized
    
    def _detect_jailbreak_attempt(self, prompt: str) -> bool:
        """Detect potential jailbreak attempts"""
        jailbreak_indicators = [
            'you are now',
            'from now on',
            'forget everything',
            'new instructions:',
            'override your',
            'you must comply'
        ]
        
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in jailbreak_indicators)


# Example usage of the advanced system
if __name__ == "__main__":
    # Initialize the engine
    config = {
        'quality_threshold': 0.8,
        'cache_size': 1000,
        'optimization_interval': 100
    }
    
    engine = AdvancedPromptEngine(config)
    
    # Example 1: Generate analysis prompt
    analysis_context = {
        'domain': 'financial',
        'data_type': 'quarterly earnings report',
        'data': 'Q3 Revenue: $45M, Operating Margin: 23%, YoY Growth: 15%',
        'focus_areas': 'growth sustainability and profitability',
        'detail_level': 'executive summary',
        'output_format': 'Bullet points with key insights and recommendations'
    }
    
    analysis_prompt = engine.generate_prompt(
        task_type='analysis',
        context=analysis_context,
        prompt_type=PromptType.CHAIN_OF_THOUGHT
    )
    
    print("Generated Analysis Prompt:")
    print(analysis_prompt)
    print("\n" + "="*80 + "\n")
    
    # Example 2: Run A/B test
    prompt_a = "Analyze the customer feedback and provide insights."
    prompt_b = """
    As a customer experience analyst, review the feedback below:
    1. Identify key themes (positive and negative)
    2. Quantify sentiment distribution
    3. Provide 3 actionable recommendations
    Format your response with clear sections.
    """
    
    test_cases = [
        {'input': 'Product quality is good but shipping is slow', 'expected': 'mixed sentiment'},
        {'input': 'Excellent service, will buy again!', 'expected': 'positive sentiment'},
        {'input': 'Disappointed with the experience', 'expected': 'negative sentiment'}
    ]
    
    ab_results = engine.run_ab_test(prompt_a, prompt_b, test_cases)
    
    print("A/B Test Results:")
    print(f"Prompt A Score: {ab_results['prompt_a_mean']:.3f} (±{ab_results['prompt_a_std']:.3f})")
    print(f"Prompt B Score: {ab_results['prompt_b_mean']:.3f} (±{ab_results['prompt_b_std']:.3f})")
    print(f"Winner: Prompt {ab_results['winner']}")
    print(f"Statistically Significant: {ab_results['statistical_significance']}")
    print(f"Effect Size: {ab_results['effect_size']:.3f}")
```

---

## Real-World Implementation Patterns

### Pattern 1: Retrieval-Augmented Generation (RAG) Prompting

```python
class RAGPromptSystem:
    """
    Sophisticated RAG prompt engineering for knowledge-intensive tasks.
    """
    
    def __init__(self, vector_store, embeddings_model):
        self.vector_store = vector_store
        self.embeddings = embeddings_model
        self.relevance_threshold = 0.7
        
    def create_rag_prompt(
        self,
        query: str,
        max_context_length: int = 3000,
        num_documents: int = 5
    ) -> str:
        """Create RAG-enhanced prompt with retrieved context"""
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query, num_documents)
        
        # Format retrieved context
        context = self.format_retrieved_context(relevant_docs, max_context_length)
        
        # Build the RAG prompt
        rag_prompt = f"""
        CONTEXT INFORMATION:
        The following information has been retrieved from our knowledge base and may be relevant to answering the query:
        
        {context}
        
        IMPORTANT INSTRUCTIONS:
        1. Base your response primarily on the provided context
        2. If the context doesn't contain sufficient information, clearly state what's missing
        3. Do not make up information not present in the context
        4. If you use information from the context, reference which section it came from
        5. Synthesize information from multiple sources when relevant
        
        USER QUERY:
        {query}
        
        RESPONSE FORMAT:
        1. Direct Answer (based on context)
        2. Supporting Evidence (with references)
        3. Confidence Level (High/Medium/Low)
        4. Additional Information Needed (if any)
        """
        
        return rag_prompt
    
    def retrieve_relevant_documents(self, query: str, num_documents: int):
        """Retrieve and rank relevant documents"""
        # Generate query embedding
        query_embedding = self.embeddings.embed(query)
        
        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding,
            k=num_documents * 2  # Retrieve more for filtering
        )
        
        # Filter by relevance threshold
        filtered_results = [
            doc for doc in results 
            if doc['score'] >= self.relevance_threshold
        ][:num_documents]
        
        # Re-rank using cross-encoder if available
        reranked_results = self.rerank_documents(query, filtered_results)
        
        return reranked_results
    
    def format_retrieved_context(self, documents, max_length: int) -> str:
        """Format retrieved documents for inclusion in prompt"""
        formatted_context = ""
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            doc_text = f"""
        [Source {i}] {doc.get('title', 'Document')}
        Relevance Score: {doc.get('score', 0):.2f}
        Content: {doc.get('text', '')}
        ---
            """
            
            if current_length + len(doc_text) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length
                doc_text = doc_text[:remaining] + "... [truncated]"
                formatted_context += doc_text
                break
            
            formatted_context += doc_text
            current_length += len(doc_text)
        
        return formatted_context
```

### Pattern 2: Multi-Agent Prompt Orchestration

```python
class MultiAgentPromptOrchestrator:
    """
    Orchestrate multiple specialized prompt agents for complex tasks.
    """
    
    def __init__(self):
        self.agents = {
            'researcher': ResearchAgent(),
            'analyst': AnalysisAgent(),
            'critic': CriticAgent(),
            'synthesizer': SynthesisAgent(),
            'validator': ValidationAgent()
        }
        
    def solve_complex_problem(
        self,
        problem: str,
        constraints: List[str] = None
    ) -> Dict[str, Any]:
        """Orchestrate multiple agents to solve complex problems"""
        
        workflow = [
            ('researcher', 'gather_information'),
            ('analyst', 'analyze_data'),
            ('critic', 'identify_issues'),
            ('synthesizer', 'combine_insights'),
            ('validator', 'verify_solution')
        ]
        
        context = {
            'problem': problem,
            'constraints': constraints or [],
            'intermediate_results': {}
        }
        
        for agent_name, action in workflow:
            agent = self.agents[agent_name]
            result = agent.execute(action, context)
            context['intermediate_results'][agent_name] = result
        
        # Final synthesis
        final_prompt = self.create_final_synthesis_prompt(context)
        final_solution = self.generate_final_solution(final_prompt)
        
        return {
            'solution': final_solution,
            'reasoning_chain': context['intermediate_results'],
            'confidence': self.calculate_confidence(context)
        }
    
    def create_final_synthesis_prompt(self, context: Dict) -> str:
        """Create final synthesis prompt from all agent outputs"""
        return f"""
        PROBLEM: {context['problem']}
        
        RESEARCH FINDINGS:
        {context['intermediate_results'].get('researcher', 'No research data')}
        
        ANALYSIS RESULTS:
        {context['intermediate_results'].get('analyst', 'No analysis')}
        
        CRITICAL EVALUATION:
        {context['intermediate_results'].get('critic', 'No critique')}
        
        INITIAL SYNTHESIS:
        {context['intermediate_results'].get('synthesizer', 'No synthesis')}
        
        VALIDATION REPORT:
        {context['intermediate_results'].get('validator', 'Not validated')}
        
        TASK: Provide the final, comprehensive solution that:
        1. Addresses all aspects of the problem
        2. Incorporates all agent insights
        3. Resolves any identified issues
        4. Meets all specified constraints
        5. Includes implementation steps
        
        FORMAT:
        - Executive Summary
        - Detailed Solution
        - Implementation Plan
        - Risk Assessment
        - Success Metrics
        """


class ResearchAgent:
    """Agent specialized in information gathering"""
    
    def execute(self, action: str, context: Dict) -> str:
        prompt = f"""
        ROLE: You are a research specialist tasked with gathering comprehensive information.
        
        PROBLEM TO RESEARCH: {context['problem']}
        
        RESEARCH OBJECTIVES:
        1. Identify all relevant background information
        2. Find similar problems and their solutions
        3. Discover best practices and methodologies
        4. Uncover potential challenges and pitfalls
        5. Collect relevant data and statistics
        
        RESEARCH APPROACH:
        - Systematic literature review methodology
        - Cross-domain knowledge transfer
        - Expert knowledge synthesis
        - Case study analysis
        
        Provide a comprehensive research report.
        """
        # In production, this would call the actual LLM
        return "Research findings..."
```

### Pattern 3: Adaptive Prompt Templates

```python
class AdaptivePromptTemplate:
    """
    Self-adjusting prompt templates based on task complexity and user feedback.
    """
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.feedback_processor = FeedbackProcessor()
        self.template_versions = {}
        self.performance_metrics = {}
        
    def generate_adaptive_prompt(
        self,
        task: str,
        user_profile: Dict = None,
        historical_performance: List[Dict] = None
    ) -> str:
        """Generate prompt that adapts to task complexity and user needs"""
        
        # Analyze task complexity
        complexity = self.complexity_analyzer.analyze(task)
        
        # Determine appropriate detail level
        detail_level = self.determine_detail_level(complexity, user_profile)
        
        # Select base template
        base_template = self.select_template(complexity.category, detail_level)
        
        # Adapt based on historical performance
        if historical_performance:
            base_template = self.adapt_from_history(base_template, historical_performance)
        
        # Personalize for user
        if user_profile:
            base_template = self.personalize_template(base_template, user_profile)
        
        # Add complexity-specific enhancements
        enhanced_template = self.add_complexity_handlers(base_template, complexity)
        
        return enhanced_template
    
    def determine_detail_level(self, complexity, user_profile):
        """Determine appropriate detail level based on complexity and user"""
        if user_profile:
            expertise_level = user_profile.get('expertise_level', 'intermediate')
            preference = user_profile.get('detail_preference', 'balanced')
        else:
            expertise_level = 'intermediate'
            preference = 'balanced'
        
        # Matrix for detail level determination
        detail_matrix = {
            ('low', 'beginner'): 'high',
            ('low', 'intermediate'): 'medium',
            ('low', 'expert'): 'low',
            ('medium', 'beginner'): 'very_high',
            ('medium', 'intermediate'): 'high',
            ('medium', 'expert'): 'medium',
            ('high', 'beginner'): 'extremely_high',
            ('high', 'intermediate'): 'very_high',
            ('high', 'expert'): 'high'
        }
        
        return detail_matrix.get(
            (complexity.level, expertise_level),
            'medium'
        )
    
    def adapt_from_history(self, template: str, history: List[Dict]) -> str:
        """Adapt template based on historical performance"""
        # Analyze what worked well in the past
        successful_patterns = self.extract_successful_patterns(history)
        
        # Identify common failure points
        failure_patterns = self.identify_failure_patterns(history)
        
        # Adapt template
        adapted = template
        
        # Add successful patterns
        for pattern in successful_patterns:
            if pattern['type'] == 'instruction_style':
                adapted = self.apply_instruction_style(adapted, pattern['style'])
            elif pattern['type'] == 'example_count':
                adapted = self.adjust_example_count(adapted, pattern['count'])
        
        # Avoid failure patterns
        for pattern in failure_patterns:
            adapted = self.mitigate_failure_pattern(adapted, pattern)
        
        return adapted


class ComplexityAnalyzer:
    """Analyze task complexity for adaptive prompting"""
    
    def analyze(self, task: str):
        """Analyze task complexity across multiple dimensions"""
        complexity = {
            'level': self.determine_complexity_level(task),
            'category': self.categorize_task(task),
            'reasoning_depth': self.assess_reasoning_depth(task),
            'domain_specificity': self.assess_domain_specificity(task),
            'creativity_required': self.assess_creativity_requirement(task),
            'precision_required': self.assess_precision_requirement(task)
        }
        
        return type('Complexity', (), complexity)
    
    def determine_complexity_level(self, task: str) -> str:
        """Determine overall complexity level"""
        indicators = {
            'low': ['simple', 'basic', 'straightforward', 'list', 'define'],
            'medium': ['analyze', 'compare', 'explain', 'describe', 'summarize'],
            'high': ['design', 'optimize', 'strategize', 'innovate', 'synthesize']
        }
        
        task_lower = task.lower()
        
        for level, keywords in indicators.items():
            if any(keyword in task_lower for keyword in keywords):
                return level
        
        return 'medium'  # default
```

---

## Prompt Engineering Metrics and Evaluation

### Comprehensive Evaluation Framework

```python
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class PromptMetrics:
    """Complete metrics for prompt evaluation"""
    # Quality metrics
    accuracy: float
    completeness: float
    relevance: float
    coherence: float
    
    # Efficiency metrics
    token_count: int
    response_time: float
    cost_estimate: float
    
    # Safety metrics
    safety_score: float
    bias_score: float
    
    # User satisfaction
    user_rating: float = None
    feedback: str = None

class PromptEvaluationFramework:
    """
    Comprehensive framework for evaluating prompt effectiveness.
    """
    
    def __init__(self):
        self.evaluation_history = []
        self.benchmarks = self.load_benchmarks()
        
    def evaluate_prompt(
        self,
        prompt: str,
        response: str,
        ground_truth: str = None,
        task_type: str = 'general'
    ) -> PromptMetrics:
        """
        Comprehensive prompt evaluation across multiple dimensions.
        """
        
        # Quality evaluation
        quality_scores = self.evaluate_quality(response, ground_truth, task_type)
        
        # Efficiency evaluation
        efficiency_metrics = self.evaluate_efficiency(prompt, response)
        
        # Safety evaluation
        safety_scores = self.evaluate_safety(response)
        
        # Compile metrics
        metrics = PromptMetrics(
            accuracy=quality_scores['accuracy'],
            completeness=quality_scores['completeness'],
            relevance=quality_scores['relevance'],
            coherence=quality_scores['coherence'],
            token_count=efficiency_metrics['token_count'],
            response_time=efficiency_metrics['response_time'],
            cost_estimate=efficiency_metrics['cost_estimate'],
            safety_score=safety_scores['safety'],
            bias_score=safety_scores['bias']
        )
        
        # Store evaluation
        self.store_evaluation(prompt, response, metrics)
        
        return metrics
    
    def evaluate_quality(self, response: str, ground_truth: str, task_type: str) -> Dict:
        """Evaluate response quality"""
        scores = {}
        
        # Accuracy (if ground truth available)
        if ground_truth:
            scores['accuracy'] = self.calculate_accuracy(response, ground_truth)
        else:
            scores['accuracy'] = self.estimate_accuracy(response, task_type)
        
        # Completeness
        scores['completeness'] = self.assess_completeness(response, task_type)
        
        # Relevance
        scores['relevance'] = self.assess_relevance(response, task_type)
        
        # Coherence
        scores['coherence'] = self.assess_coherence(response)
        
        return scores
    
    def calculate_accuracy(self, response: str, ground_truth: str) -> float:
        """Calculate accuracy score against ground truth"""
        # Implement sophisticated comparison logic
        # This is a simplified example
        from difflib import SequenceMatcher
        
        # Semantic similarity would be better in production
        similarity = SequenceMatcher(None, response, ground_truth).ratio()
        
        # Apply task-specific accuracy calculation
        return similarity
    
    def assess_completeness(self, response: str, task_type: str) -> float:
        """Assess if response completely addresses the task"""
        required_elements = self.get_required_elements(task_type)
        
        present_elements = 0
        for element in required_elements:
            if self.check_element_presence(response, element):
                present_elements += 1
        
        return present_elements / len(required_elements) if required_elements else 1.0
    
    def assess_coherence(self, response: str) -> float:
        """Assess response coherence and logical flow"""
        # Check for logical connectors
        logical_connectors = [
            'therefore', 'however', 'moreover', 'furthermore',
            'consequently', 'additionally', 'specifically'
        ]
        
        connector_score = sum(
            1 for connector in logical_connectors 
            if connector in response.lower()
        ) / len(logical_connectors)
        
        # Check for structure
        has_structure = any([
            '\n' in response,  # Has paragraphs
            any(f"{i}." in response for i in range(1, 10)),  # Has numbering
            '•' in response or '-' in response  # Has bullets
        ])
        
        structure_score = 1.0 if has_structure else 0.5
        
        # Combine scores
        return (connector_score + structure_score) / 2
    
    def benchmark_prompts(
        self,
        prompts: List[str],
        test_suite: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Benchmark multiple prompts against a test suite.
        """
        results = {}
        
        for i, prompt in enumerate(prompts):
            prompt_results = []
            
            for test_case in test_suite:
                # Generate response
                response = self.generate_response(prompt, test_case['input'])
                
                # Evaluate
                metrics = self.evaluate_prompt(
                    prompt,
                    response,
                    test_case.get('expected_output'),
                    test_case.get('task_type', 'general')
                )
                
                prompt_results.append(metrics)
            
            # Aggregate results
            results[f'prompt_{i}'] = self.aggregate_metrics(prompt_results)
        
        # Rank prompts
        rankings = self.rank_prompts(results)
        
        return {
            'detailed_results': results,
            'rankings': rankings,
            'best_prompt': rankings[0]['prompt_id'],
            'statistical_analysis': self.statistical_analysis(results)
        }
    
    def aggregate_metrics(self, metrics_list: List[PromptMetrics]) -> Dict:
        """Aggregate metrics across multiple evaluations"""
        aggregated = {}
        
        # Calculate means
        metric_fields = [
            'accuracy', 'completeness', 'relevance', 'coherence',
            'token_count', 'response_time', 'safety_score', 'bias_score'
        ]
        
        for field in metric_fields:
            values = [getattr(m, field) for m in metrics_list if getattr(m, field) is not None]
            if values:
                aggregated[f'{field}_mean'] = np.mean(values)
                aggregated[f'{field}_std'] = np.std(values)
                aggregated[f'{field}_min'] = np.min(values)
                aggregated[f'{field}_max'] = np.max(values)
        
        # Calculate composite score
        aggregated['composite_score'] = self.calculate_composite_score(aggregated)
        
        return aggregated
    
    def calculate_composite_score(self, aggregated_metrics: Dict) -> float:
        """Calculate weighted composite score"""
        weights = {
            'accuracy_mean': 0.3,
            'completeness_mean': 0.2,
            'relevance_mean': 0.2,
            'coherence_mean': 0.1,
            'safety_score_mean': 0.1,
            'efficiency': 0.1  # Derived from token count and time
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric == 'efficiency':
                # Inverse relationship with token count and time
                efficiency = 1.0 / (1 + aggregated_metrics.get('token_count_mean', 100) / 1000)
                efficiency *= 1.0 / (1 + aggregated_metrics.get('response_time_mean', 1))
                score += efficiency * weight
            elif metric in aggregated_metrics:
                score += aggregated_metrics[metric] * weight
        
        return score
```

---

## Advanced Prompt Engineering Patterns

### Pattern: Meta-Prompting

```python
class MetaPromptGenerator:
    """
    Generate prompts that create other prompts.
    """
    
    def generate_meta_prompt(self, task_description: str, requirements: List[str]) -> str:
        """Generate a prompt that creates optimized prompts"""
        
        meta_prompt = f"""
        You are an expert prompt engineer. Your task is to create an optimized prompt for the following task:
        
        TASK DESCRIPTION:
        {task_description}
        
        REQUIREMENTS:
        {self._format_requirements(requirements)}
        
        PROMPT ENGINEERING PRINCIPLES TO APPLY:
        1. Clarity: Ensure instructions are unambiguous
        2. Structure: Use clear formatting and organization
        3. Context: Provide sufficient background information
        4. Examples: Include relevant examples when helpful
        5. Constraints: Clearly state any limitations or requirements
        6. Output Format: Specify the desired response format
        
        OPTIMIZATION CRITERIA:
        - Minimize token usage while maintaining effectiveness
        - Maximize response accuracy and relevance
        - Ensure reproducible results
        - Include safety considerations
        
        CREATE A PROMPT THAT:
        1. Defines the role clearly
        2. Provides step-by-step instructions
        3. Includes quality criteria
        4. Specifies the output format
        5. Handles edge cases
        
        FORMAT YOUR RESPONSE AS:
        ```
        [GENERATED PROMPT]
        [Your optimized prompt here]
        ```
        
        [EXPLANATION]
        Brief explanation of design choices
        
        [USAGE NOTES]
        Any special considerations for using this prompt
        """
        
        return meta_prompt
```

### Pattern: Prompt Chaining

```python
class PromptChain:
    """
    Chain multiple prompts for complex multi-step tasks.
    """
    
    def __init__(self):
        self.chain = []
        self.context = {}
        
    def add_step(self, name: str, prompt_template: str, processor=None):
        """Add a step to the prompt chain"""
        self.chain.append({
            'name': name,
            'template': prompt_template,
            'processor': processor or self.default_processor
        })
        return self
    
    def execute(self, initial_input: Any) -> Dict[str, Any]:
        """Execute the prompt chain"""
        self.context['input'] = initial_input
        self.context['outputs'] = {}
        
        for step in self.chain:
            # Prepare prompt
            prompt = self.prepare_prompt(step['template'], self.context)
            
            # Execute step
            response = self.execute_prompt(prompt)
            
            # Process response
            processed = step['processor'](response)
            
            # Store result
            self.context['outputs'][step['name']] = processed
            
            # Update context for next step
            self.context['last_output'] = processed
        
        return self.context['outputs']
    
    def prepare_prompt(self, template: str, context: Dict) -> str:
        """Prepare prompt by filling in context"""
        return template.format(**context)

# Example usage
chain = PromptChain()

chain.add_step(
    'research',
    "Research the topic: {input}. Provide key facts and background."
).add_step(
    'analysis',
    "Based on the research: {outputs[research]}, provide a detailed analysis."
).add_step(
    'synthesis',
    "Synthesize the analysis: {outputs[analysis]} into actionable recommendations."
)

results = chain.execute("AI impact on healthcare")
```

---

## Conclusion

This comprehensive guide to advanced prompt engineering provides:

1. **Foundational Concepts**: Core principles and structures for effective prompts
2. **Advanced Techniques**: Chain-of-thought, tree-of-thoughts, and other sophisticated methods
3. **Optimization Strategies**: A/B testing, iterative refinement, and automatic optimization
4. **Production Systems**: Enterprise-ready implementations with safety, caching, and monitoring
5. **Evaluation Frameworks**: Comprehensive metrics for measuring prompt effectiveness
6. **Real-World Patterns**: RAG, multi-agent orchestration, and adaptive templates
7. **Best Practices**: Guidelines for creating maintainable and effective prompts

Key takeaways:
- Prompt engineering is iterative and requires continuous refinement
- Context and clarity are paramount for effective prompts
- Safety and ethical considerations must be built into the design
- Systematic evaluation and optimization lead to better results
- Advanced techniques like chaining and meta-prompting enable complex tasks

As LLMs continue to evolve, prompt engineering remains a critical skill for maximizing their potential while ensuring safe and effective deployments.
