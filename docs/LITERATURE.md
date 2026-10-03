# Did the Harness Come First? A Literature Review for the Evo-Harness Project

**Prepared for:** the evo-harness research block (Solana "Dy")
**Date:** 2026-10-02
**Question driving this review:** *"Did the harness come first, or the brain?"* — i.e., should AI agents evolve their tooling, scaffolding, and surrounding software (the "harness") under evolutionary pressure while the model (the "brain") stays fixed?

**Abstract.** This review surveys four bodies of literature bearing on the evo-harness hypothesis: (a) biology and cognitive science, where nerve nets, distributed nervous systems, and embodied computation all predate and out-scale centralized brains; (b) AI research on agents that invent their own tools and skills, showing that tool/harness creation is learnable and compounds capability; (c) the rapidly growing field of self-improving and evolving agents, where the editable, evolvable object is overwhelmingly the harness (prompts, workflows, code, tools) rather than the model weights; and (d) the industry discourse that has recently converged on "the harness is the product." Taken together, these literatures suggest the answer to Dy's question is *the harness came first — and most of the action still is in the harness*. The review closes with four pre-registerable hypotheses (H1–H4) for the evo-harness experiments.

---

## (a) Biology and cognitive science: sensing, acting, and offloading computation before the brain existed

### Nerve nets before brains: the Hydra and the cnidarians

The first nervous systems were not brains at all. The dominant evolutionary picture places diffuse, decentralized nerve nets — as seen today in *Hydra*, jellyfish, and other cnidarians — as the ancestral nervous system, hundreds of millions of years before any centralized brain. Reporting on the first whole-animal calcium imaging of the hydra nervous system (Dupre & Yuste), *New Scientist* describes hydra as possessing "the most basic nervous system in nature, a nerve net in which neurons spread throughout its body," with distinct, dedicated circuits for feeding, contraction, and light sensing — and no neuron serving more than one circuit. The journal literature is explicit: "A prevailing view of the cnidarian nervous system is that the neural network is simple and diffuse throughout the animal body as can be observed in the freshwater polyp *Hydra*" (Galliot et al., "Cnidarians and the evolutionary origin of the nervous system," *Development, Growth & Differentiation*). Centralization of nerve cords, then cephalization (concentration of neurons at the anterior into a brain), came later. **The harness — a body-wide sensing-and-acting substrate — literally preceded the brain in animal evolution.**

### Decentralized control: the octopus and its nine brains

If cnidarians show that the harness came first, the octopus shows that the brain can stay a minority partner even at the peak of invertebrate intelligence. As a *Nature* review ("The autonomous arms of the octopus") documents, octopuses have roughly 500 million neurons — but only 40–45 million in the central brain; 120–180 million sit in the optic lobes, and "two-thirds of the neurons (~330 million) are in the octopus's eight arms." Severed arms continue to reach, grasp, and avoid noxious stimuli without the central brain, and arm nerve cords exchange information via a neural ring that bypasses the brain entirely. Evolution built the highest-bandwidth, most autonomous control systems in the body, not in the head. For evo-harness, the octopus is the canonical existence proof of **competence under a fixed, modest central controller achieved by growing the peripheral harness** — exactly the architecture (fixed LLM "brain," evolvable tool harness) this project tests.

### Morphological computation: the body as part of the computer

Pfeifer and Bongard's *How the Body Shapes the Way We Think* (MIT Press, 2007) makes the engineering case: intelligence is an emergent property of the complete agent — body morphology, materials, sensors, environment, and nervous system in ecological balance. Their "cheap design" principle and the concept of *morphological computation* — the body performing processes that otherwise would have to be controlled by the brain — argue that cognitive workload should be pushed out of the central controller into the periphery wherever the morphology allows it. Their design principle of "ecological balance" (matching complexity across sensory, motor, and neural systems) is directly applicable: when the brain (LLM) is fixed, evolution must act on morphology (tools, scaffolding) to restore balance. This is the same instinct as the Müller & Hoffmann treatment of morphological computation, which the book's lineage anticipates.

Brooks' subsumption architecture ("A Robust Layered Control System for a Mobile Robot," 1986) made the same point in robotics decades earlier: stack layers of behavioral competence such that "control is layered with higher levels subsuming the roles of lower level layers when they wish to take control," with lower layers forming "a complete operational control system" even if severed from higher layers. An LLM agent with an evolvable tool harness is, architecturally, a subsumption system: the harness layers provide complete behavioral competence, and the model's job is steering, suppression, and modulation — not raw execution.

### The extended mind and basal cognition

Clark and Chalmers' "The Extended Mind" (*Analysis*, 1998) argued via the parity principle that reliably accessible external processes — notebooks, tools, environments — count as part of the cognitive system itself. An agent's tool library is the most literal instantiation of the extended-mind thesis: the harness is not "around" the cognition; under the parity principle, it *is* cognition.

Levin's basal-cognition program pushes the point deeper into biology. In "Technological Approach to Mind Everywhere" (*Frontiers in Systems Neuroscience*, 2022) and "Bioelectric networks: the cognitive glue enabling evolutionary scaling from physiology to mind" (*Animal Cognition*, 2023), Levin shows that goal-directed information processing — memory, learning, decision-making — occurs in cells, tissues, and aneural organisms via pre-neural bioelectric networks, long before and independently of neurons. Cognition, in this view, is scale-agnostic problem-solving, and the bioelectric "harness" that couples cells into anatomical goals is evolutionarily older than any brain. **The recurring biological answer is the same: the infrastructure for goal-directed behavior came first; the centralized controller is a late, thin layer on top.**

> **Synthesis for Dy's question.** Biology answers "the harness came first" unambiguously: nerve nets preceded brains, two-thirds of the octopus's neurons still live in its arms, morphological computation offloads control into bodies, and basal cognition operates without neurons at all. The design implication is the "octopus architecture": a fixed, modest central controller becomes far more capable when evolution is allowed to grow the periphery. This is the biological license for evo-harness.

---

## (b) AI tool and skill evolution: agents that grow their own harness

### Voyager: the founding demonstration

Voyager (Wang et al., 2023, "Voyager: An Open-Ended Embodied Agent with Large Language Models") is the landmark result for harness evolution: an agent that keeps its model (GPT-4) fixed and grows an ever-expanding *skill library* of executable code, driven by an automatic curriculum and iterative prompting with environment feedback. The results — 3.3× more unique items, 15.3× faster tech-tree milestones than prior SOTA in Minecraft, and skill transfer to new worlds — were achieved with zero weight updates. Voyager is, in essence, the first large-scale proof of the evo-harness hypothesis in a sandbox: **fix the brain, evolve the harness, and capability compounds.**

### Tool creation: CREATOR, CRAFT, ToolMaker

Qian et al. (2023, "CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation," EMNLP Findings) showed LLMs can create their own tools (documentation + code) to split abstract reasoning from concrete execution, outperforming chain-of-thought, program-of-thought, and tool-using baselines on MATH and TabMWP. CRAFT (Yuan et al., 2024, ICLR) built reusable, validated toolsets from generated code snippets, retrieved at inference time. ToolMaker (Wölflein et al., 2025, ACL) closes a bigger loop: an agentic framework that autonomously converts scientific papers-with-code into LLM-compatible tools, correctly implementing 80% of a 15-task benchmark with over 100 unit tests. The trajectory from CREATOR to ToolMaker is a steady widening of what the harness can absorb: from one-off code tools, to curated toolsets, to whole repositories compiled into tools.

### Tool use as a generable, refinable artifact

Two complementary results fill out the picture. ToolGen (Wang et al., 2025, ICLR) shows the other extreme: folding the tool interface itself into the model by representing each of 47,000 tools as a unique token and training the model to generate tool calls as next-token prediction — i.e., the harness can also be pushed *into* the brain via parameter training, at the cost of losing the fixed-brain, modular-updatability that evo-harness wants to preserve. AvaTaR (Wu et al., 2024, NeurIPS, "AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning") optimizes tool-using prompts by contrastively reasoning over positive and negative examples, improving Hit@1 by ~14% relative on retrieval tasks — a harness-level refinement loop (prompts about tool use) that needs no weight changes. And Ruan et al.'s TPTU (2023, "Large Language Model-based AI Agents for Task Planning and Tool Usage") framed tool *selection, creation, and execution* as a core agent competency alongside task planning, a framing the whole subsequent literature has adopted. MetaAgent (Qian & Liu, 2025) extends the idea to tool *meta-learning* — a self-evolving agent that learns how to learn tools.

> **Synthesis for Dy's question.** The AI literature mirrors the biology: capability gains are routinely achieved by growing, curating, and refining the tool harness while the model is held fixed. Voyager is the flagship case; CREATOR/CRAFT/ToolMaker show the "tool creation" reflex can itself be automated; AvaTaR and MetaAgent show the harness can be optimized and meta-learned. Only ToolGen takes the opposite route (absorb the harness into the weights), and its cost — retraining, loss of modularity — is precisely the argument for evolving the harness instead.

---

## (c) Self-improving and evolving agents: evolution acts on the harness, not the weights

A striking regularity across the self-improving-agents literature is *what* gets evolved. Almost never the weights; almost always the harness — prompts, workflows, code, skills, tools, and architectures around the model.

### Open-ended harness evolution: DGM, ADAS, AFlow, AgentSquare, EvoAgent

The Darwin Gödel Machine (Zhang et al., 2025, "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents") is the strongest recent statement: a coding agent that rewrites its own Python codebase (tools, workflow, prompting — explicitly not model weights), validates each modification empirically, and keeps an *archive* of past variants to enable open-ended exploration. SWE-bench Verified rises from 20% to 50%; critically, the ablation shows the archive (not just greedy hill-climbing) is what prevents the agent from getting stuck. Its direct ancestor, ADAS (Hu, Lu & Clune, 2024, "Automated Design of Agentic Systems," ICLR 2025), framed the general principle: since Python is Turing-complete, searching over *code that defines the agent* can in principle express any agent design, and its "Meta Agent Search" discovers agent programs that beat hand-designed ones. AFlow (Jiayi Zhang et al., 2024, ICLR 2025 Oral, "AFlow: Automating Agentic Workflow Generation") applies MCTS over executable workflow graphs, showing agent orchestration structure is itself an optimizable substrate. AgentSquare (Shang et al., 2024, "AgentSquare: Automatic LLM Agent Search in Modular Design Space") modularizes agents into planning/reasoning/tool-use/memory modules with uniform interfaces and searches via module evolution and recombination (+17.2% average gain over hand-crafted designs). EvoAgent (Siyu Yuan et al., 2024, NAACL 2025, "EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms") treats existing agent frameworks as initial individuals and applies evolutionary operators to grow multi-agent systems automatically.

### Prompt and program evolution: Promptbreeder, OPRO, DSPy, TextGrad, GEPA

An older, equally important lineage evolves the *textual* layer of the harness. Promptbreeder (Fernando et al., 2023, "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution") evolves a population of task-prompts *and* the mutation-prompts that mutate them — self-referential improvement of the improvement machinery, beating CoT, Plan-and-Solve, APE, and OPRO on arithmetic and commonsense benchmarks with PaLM 2-L. OPRO (Yang et al., 2023, "Large Language Models as Optimizers," ICLR 2024) showed the LLM itself can serve as a gradient-free optimizer, reading a score-sorted trajectory of past attempts and proposing better prompts (+8% GSM8K, up to ~+50% on Big-Bench Hard over human-written prompts). DSPy (Khattab et al., 2023, "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines," ICLR 2024) reframed the whole problem as compilation: declare LM programs, let teleprompters optimize them against metrics. TextGrad (Yuksekgonul et al., 2024, "TextGrad: Automatic 'Differentiation' via Text") gave the field its autograd engine for natural language — textual gradients backpropagated through composed systems. GEPA (Agrawal et al., 2025, "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") pushes textual evolution to its current frontier: reflective, Pareto-frontier-based evolution of prompts that beats GRPO (RL) by 6% on average while using 35× fewer rollouts — i.e., **evolving the harness in language-space is now competitive with updating weights via RL**, at a fraction of the sample cost.

### Inference-time and generalist self-evolution: Reflexion, Mind Evolution, Alita, LLaMEA

Reflexion (Shinn et al., 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023) demonstrated that verbal self-critique stored in episodic memory improves trial-to-trial performance (HumanEval 91% pass@1 vs. 80% GPT-4 baseline) with no weight updates — the minimal harness-evolution loop. Mind Evolution (Lee et al., 2025, "Evolving Deeper LLM Thinking") scales inference-time compute with an evolutionary strategy that generates, recombines, and refines candidate responses, solving >98% of TravelPlanner and Natural Plan instances and beating Best-of-N and Sequential Revision at matched cost. Alita (Qiu et al., 2025, "Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution") starts from a deliberately minimal agent and lets it construct, refine, and reuse capabilities by generating task-specific Model Context Protocol (MCP) tools from open source — maximal self-evolution of the harness from a minimal core. LLaMEA (van Stein & Bäck, *IEEE Transactions on Evolutionary Computation*; "LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics") rounds out the picture: LLMs as mutation/crossover operators inside an evolutionary loop, generating novel algorithms that beat state-of-the-art metaheuristics on BBOB — evolution *of programs*, by programs, with fixed models.

> **Synthesis for Dy's question.** The entire self-improving-agents literature is, read one way, a decades-long answer to "harness or brain?": every successful system evolves the harness — prompts (Promptbreeder, OPRO, GEPA), workflows (AFlow), agent code (DGM, ADAS), modules (AgentSquare), tools/skills (Voyager, Alita), memories (Reflexion) — while holding weights fixed. The one consistent failure mode is greedy keep-if-better hill-climbing; the consistent fix (DGM, ADAS, GEPA's Pareto frontier, Promptbreeder's population) is archives/diversity. These are the mechanisms evo-harness should steal directly.

---

## (d) The harness discourse: industry converges on the same answer

The phrase "the harness is the product" circulates widely in 2025–2026 agent-engineering writing, but the canonical, attributable formulations are more specific and more useful:

- **Mitchell Hashimoto, "My AI Adoption Journey" (Feb 2026)** coined the industry's working formula — **"Agent = Model + Harness"** — with the operational maxim: "Anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again." (https://mitchellh.com/writing/my-ai-adoption-journey)
- **Anthropic's engineering writing** frames the relationship crisply — the model as the engine and the harness as the car around it — in Prithvi Rajasekaran's "Harness Design for Long-Running Application Development" (2026), which shows harness components being systematically added and removed as model generations change what the scaffolding must do. The foundational "Building effective agents" (Schluntz & Zhang, Dec 2024) separates *workflows* (human-fixed code paths) from *agents* (model-directed control flow) and argues most production value lives in the composable scaffolding — prompts, tools, retrieval, memory — around the model. (https://www.anthropic.com/engineering/harnessdesignlongrunningapps ; https://www.anthropic.com/research/building-effective-agents)
- **OpenAI's "Harness engineering" page** (Feb 2026, by Ryan Lopopolo) institutionalizes the discipline: designing environments, feedback loops, constraints, and control systems that make agents reliable at scale — including the striking report of an internal product built with zero hand-written lines of code, where the engineers' job shifted entirely to environment design and intent specification. (https://openai.com/index/harness-engineering/)
- **Martin Fowler and Birgitta Boeckeler (Thoughtworks, Apr 2026)** formalized a taxonomy of harness components: context engineering, architectural constraints, and entropy management. (https://martinfowler.com/articles/harness-engineering.html)
- **Karpathy** supplies the era framing: the "Software Is Changing (Again)" keynote (YC AI Startup School, June 2025) names Software 3.0 — prompts-as-programs — and his widely adopted "context engineering" endorsement ("the delicate art and science of filling the context window with just the right information for the next step") recasts harness-building as *the* core engineering discipline of the agent era. Anthropic's "Effective context engineering for AI agents" (Sep 2025) codifies it: curating the smallest high-signal token set that maximizes the chance of the desired outcome. (https://www.ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again ; https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Shunyu Yao, "The Second Half" (Apr 2025)** argues the field has entered the second half of the agent game, where "evaluation becomes more important than training" — i.e., the bottleneck is no longer model capability but defining, measuring, and selecting for agent behavior. That is a harness-side bottleneck: selection pressure needs an environment and a fitness function, not a bigger brain. (https://ysymyth.github.io/The-Second-Half/)

Note on attribution: the exact slogan "the model is not the product, the harness is the product" is industry paraphrase, not a line traceable to a single canonical author (it is not Karpathy's). The attributable claims are Hashimoto's "Agent = Model + Harness" and Anthropic's "model is the engine, harness is the car." Both say the same thing: value accrues in the layer around the model.

> **Synthesis for Dy's question.** Industry practice has independently rediscovered the biological answer: the model is a component; the product, the reliability, and the compounding advantage all live in the harness. The discourse even converges on what evo-harness needs — evolvable context engineering (Fowler/Boeckeler), selection via evaluation (Yao's "Second Half"), and systematic mistake-to-structure conversion (Hashimoto's maxim) — which reads like a recipe for an evolutionary harness loop.

---

## Key takeaways for our experiments

If the harness came first and still does most of the work, then an evolutionary experiment should fix the model and apply selection pressure to the harness, with the mechanisms the literature has already validated:

- **H1 — Harness evolution beats prompt-only evolution.** A fixed LLM with an evolvable tool/skill harness (add/prune/refine tools, per Voyager and Alita) will improve faster and further on a multi-step task suite than the same model with only its prompts evolved (per GEPA/OPRO), because tools compound: each kept tool changes what future reasoning can reach. *Prediction:* tool-harness populations show superlinear early gains vs. prompt-only populations.
- **H2 — Archives beat greedy hill-climbing.** Following DGM's ablation and ADAS/GEPA's keep-all/Pareto designs, a selection mechanism that retains a diverse archive of past harness variants (including currently-suboptimal ones) will avoid the "ratchet trap" where a poor modification blocks later progress. *Prediction:* greedy keep-if-better stagnates; archive-based selection continues improving.
- **H3 — Morphological pressure finds octopus-like architectures.** Under a task suite that rewards speed and parallelism, evolution will push work out of the model and into the harness — precomputed tools, cached subroutines, decomposed sub-agents — mirroring how octopus arms absorbed two-thirds of the neurons. *Prediction:* evolved harnesses use fewer model calls per task and more tool-internal computation than hand-designed baselines.
- **H4 — Evaluation is the bottleneck (Yao's "Second Half").** The quality of the fitness function and task suite will bound harness evolution more than the quality of the mutation operator. *Prediction:* a richer, task-diverse evaluation environment yields better final harnesses than a stronger mutation LLM on a thin benchmark.

**Pre-registration note:** these hypotheses should be locked before the first evolution run, with the task suite, budget, and selection mechanisms documented per-hypothesis in the experiment configs.

---

## References

### (a) Biology & cognitive science

1. Dupre, C. & Yuste, R. — Whole-brain calcium imaging of the hydra nerve net (reported in *New Scientist*, "Entire nervous system of an animal recorded for the first time," 2017). [link](https://www.newscientist.com/article/2127625-entire-nervous-system-of-an-animal-recorded-for-the-first-time/)
2. Galliot, B. et al. — "Cnidarians and the evolutionary origin of the nervous system," *Development, Growth & Differentiation* (2009). [link](https://onlinelibrary.wiley.com/doi/10.1111/j.1440-169X.2009.01103.x)
3. "The autonomous arms of the octopus," *Lab Animal / Nature* (2014) — ~2/3 of the octopus's ~500M neurons reside in the arms. [link](https://www.nature.com/articles/laban.615)
4. Pfeifer, R. & Bongard, J. — *How the Body Shapes the Way We Think: A New View of Intelligence*, MIT Press (2007). [link](https://www.betterworldbooks.com/product/detail/how-the-body-shapes-the-way-we-think-a-new-view-of-intelligence-9780262537421)
5. Brooks, R. A. — "A Robust Layered Control System for a Mobile Robot," *IEEE Journal of Robotics and Automation* 2(1):14–23 (1986). [link](https://faculty.washington.edu/minster/bio_inspired_robotics/research_articles/brooks_robust_layered_control_robot_ieeetransrobotautomat1986.pdf)
6. Clark, A. & Chalmers, D. — "The Extended Mind," *Analysis* 58(1):7–19 (1998). [link](https://doi.org/10.1093/analys/58.1.7)
7. Levin, M. — "Technological Approach to Mind Everywhere: An Experimentally-Grounded Framework for Understanding Diverse Bodies and Minds," *Frontiers in Systems Neuroscience* 16:768201 (2022). [link](https://www.medscape.com/medline/abstract/35401131)
8. Levin, M. — "Bioelectric networks: the cognitive glue enabling evolutionary scaling from physiology to mind," *Animal Cognition* 26(6):1865–1891 (2023). [link](https://doi.org/10.1007/s10071-023-01780-3)

### (b) Tool & skill evolution

9. Wang, G. et al. — "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023). [link](https://arxiv.org/abs/2305.16291)
10. Qian, C. et al. — "CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation," EMNLP Findings (2023). [link](https://arxiv.org/abs/2305.14318)
11. Yuan, L. et al. — "CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets," ICLR (2024). [link](https://arxiv.org/abs/2309.17428)
12. Wölflein, G. et al. — "LLM Agents Making Agent Tools" (ToolMaker), ACL (2025). [link](https://arxiv.org/abs/2502.11705)
13. Wang, R. et al. — "ToolGen: Unified Tool Retrieval and Calling via Generation," ICLR (2025). [link](https://arxiv.org/abs/2410.03439)
14. Wu, S. et al. — "AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning," NeurIPS (2024). [link](https://arxiv.org/abs/2406.11200)
15. Ruan, J. et al. — "TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage" (2023). [link](https://arxiv.org/abs/2308.03427)
16. Qian, H. & Liu, Z. — "MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning" (2025). [link](https://arxiv.org/abs/2508.00271)

### (c) Self-improving & evolving agents

17. Zhang, J. et al. — "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents" (2025). [link](https://arxiv.org/abs/2505.22954)
18. Hu, S., Lu, C. & Clune, J. — "Automated Design of Agentic Systems" (ADAS), ICLR (2025). [link](https://arxiv.org/abs/2408.08435)
19. Fernando, C. et al. — "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution" (2023). [link](https://arxiv.org/abs/2309.16797)
20. Yang, C. et al. — "Large Language Models as Optimizers" (OPRO), ICLR (2024). [link](https://arxiv.org/abs/2309.03409)
21. Khattab, O. et al. — "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines," ICLR (2024). [link](https://arxiv.org/abs/2310.03714)
22. Yuksekgonul, M. et al. — "TextGrad: Automatic 'Differentiation' via Text" (2024). [link](https://arxiv.org/abs/2406.07496)
23. Agrawal, L. A. et al. — "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (2025). [link](https://arxiv.org/abs/2507.19457)
24. Zhang, J. et al. — "AFlow: Automating Agentic Workflow Generation," ICLR Oral (2025). [link](https://arxiv.org/abs/2410.10762)
25. Yuan, S. et al. — "EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms," NAACL (2025). [link](https://arxiv.org/abs/2406.14228)
26. Shang, Y. et al. — "AgentSquare: Automatic LLM Agent Search in Modular Design Space" (2024). [link](https://arxiv.org/abs/2410.06153)
27. Qiu, J. et al. — "Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution" (2025). [link](https://arxiv.org/abs/2505.20286)
28. van Stein, N. & Bäck, T. — "LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics," *IEEE TEVC* (2025). [link](https://arxiv.org/abs/2405.20132)
29. Shinn, N. et al. — "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS (2023). [link](https://arxiv.org/abs/2303.11366)
30. Lee, K.-H. et al. — "Evolving Deeper LLM Thinking" (Mind Evolution) (2025). [link](https://arxiv.org/abs/2501.09891)

### (d) The harness discourse

31. Hashimoto, M. — "My AI Adoption Journey" (Feb 2026) — "Agent = Model + Harness." [link](https://mitchellh.com/writing/my-ai-adoption-journey)
32. OpenAI — "Harness engineering: leveraging Codex in an agent-first world" (2026). [link](https://openai.com/index/harness-engineering/)
33. Fowler, M. & Boeckeler, B. — "Harness engineering," Thoughtworks (Apr 2026). [link](https://martinfowler.com/articles/harness-engineering.html)
34. Schluntz, E. & Zhang, B. — "Building effective agents," Anthropic (Dec 2024). [link](https://www.anthropic.com/research/building-effective-agents)
35. Anthropic Engineering — "Effective context engineering for AI agents" (Sep 2025). [link](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
36. Karpathy, A. — "Software Is Changing (Again)," YC AI Startup School keynote (June 2025). [link](https://www.ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again)
37. Karpathy, A. — "Context engineering is the delicate art and science of filling the context window with just the right information for the next step" (X, June 2025). [link](https://x.com/karpathy/status/1937902205765607626)
38. Yao, S. — "The Second Half" (Apr 2025). [link](https://ysymyth.github.io/The-Second-Half/)
39. Rajasekaran, P. — "Harness Design for Long-Running Application Development," Anthropic Engineering (2026). [link](https://www.anthropic.com/engineering/harnessdesignlongrunningapps)

*All citations verified via web search on 2026-10-02; any work that could not be verified was dropped (see verification notes in the experiment README).*
