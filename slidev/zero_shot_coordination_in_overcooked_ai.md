---
theme: academic
layout: cover
class: text-white
coverAuthor: 胡逸同
coverAuthorUrl: https://yitong-hu.metattri.com
coverBackgroundUrl: https://lsky.metattri.com/i/2024/01/19/65aa2a5e7ddab.gif
coverBackgroundSource: Overcooked-AI
coverBackgroundSourceUrl: https://github.com/HumanCompatibleAI/overcooked_ai
download: https://wiederholung.github.io/zero_shot_coordination_in_overcooked_ai
export:
  format: pdf
  timeout: 30000
  dark: false
  withClicks: true
  withToc: true
hideInToc: true
fonts:
  local: Montserrat, Roboto Mono, Roboto Slab # local fonts are used for legal reasons for deployment to https://slidev-theme-academic.alexeble.de and only set up for the example project, remove this line for your project to automatically have fonts imported from Google
themeConfig:
  paginationX: r
  paginationY: t
  paginationPagesDisabled: [1]
title: Zero-Shot Coordination in Overcooked-AI
info: |
  # slidev-theme-academic

  Created and maintained by [Alexander Eble](https://www.alexeble.de).

  - [GitHub](https://github.com/alexanderdavide/slidev-theme-academic)
  - [npm](https://www.npmjs.com/package/slidev-theme-academic)

  slidev-theme-academic is licensed under [MIT](https://github.com/alexanderdavide/slidev-theme-academic/blob/master/LICENSE).

  <ul>
    <li>
      <a href="https://www.alexeble.de/impressum/" target="_blank">Legal information of this website</a>
    </li>
    <li>
      <a href="https://www.alexeble.de/datenschutz/" target="_blank">Privacy policy of this website</a>
    </li>
  </ul>
---

# Zero-Shot Coordination in Overcooked-AI

<Pagination classNames="text-gray-300" />

---
hideInToc: true
---

# 目录

<Toc maxDepth = 2 /> 

---
layout: center
class: "text-center"
---

# Overcooked-AI

---

## 简介

[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) 是由 UC Baerkeley [CHAI](https://humancompatible.ai/about/) 团队开发的 benchmark 环境，旨在通过 [Overcooked](http://www.ghosttowngames.com/overcooked/) 游戏，评估各算法在 **human-AI 完全合作**任务中的性能。

在 Overcooked-AI 中，2 个玩家需要**合作**完成 `取食材-移动食材-入锅-装盘-上菜` 一系列任务，获得团队得分。Agents 需要学习地图导航、物体交互和上菜，同时注意与伙伴的协调，属于 common-payoff game。

**环境**：

- 2 agents，agents pair = $[A_0,A_1], A_i \in [AI, Human]$
- 5 种不同的布局，各有不同的地形和物体分布
- 可交互物体 = [洋葱，盘子，锅，台面，上菜区]，环境会无限生成洋葱和盘子

![layout](https://lsky.metattri.com/i/2024/01/19/65aa2a5e7ddab.gif)

---
layout: figure-side
figureCaption: Human (Green Hat) v.s. AI (Blue Hat) in Coordination Ring
figureUrl: https://lsky.metattri.com/i/2024/01/24/65b0ca8512b73.gif
---

**Agents**：

- 动作空间 = [上、下、左、右移动，啥也不干，交互]
  > 交互：对个物体有不同的操作，例如：将洋葱入锅，拿盘子盛锅里的汤，将盛好汤的盘子放在上菜区，或把洋葱/盘子暂存至台面
- 可完全观测环境（MDP），或泛化到可部分观测环境（POMDP）

**任务**：

- `将 3 个洋葱放入锅中 - 煮 20 timesteps - 将汤装入盘子 - 将盘子放在上菜区`
- **上菜才能得分**，有时间限制
- 只有团队得分，无个人得分

---

**布局**：

![layout](https://lsky.metattri.com/i/2024/01/19/65aa2a5e7ddab.gif)

不同的布局要求不同的协作策略，从左到右，分别是：

1. Cramped Room：提供低级协调挑战，因空间限制，代理很容易相撞。
2. Asymmetric Advantages：测试玩家是否可以选择发挥自身优势的高级策略。
3. Coordination Ring：玩家必须协调，才能在布局的左下角与右上角之间移动。
4. Forced Coordination：消除了碰撞协调问题，强迫玩家发展高级联合策略，因为单个玩家无法独自上菜。
5. Counter Circuit：涉及隐式的协调策略，洋葱经柜台传递至锅中，而不是绕道携带。

---

**使用 Multi-Agent MDP 表达游戏过程** [^UtilityLearningHumans2019]：

A multi-agent MDP is defined by a tuple $\langle S, \alpha, \{A_{i \in \alpha}\}, \mathcal{T}, R \rangle$:

- $S$ is a finite set of states, and $R : S \to \mathbb{R}$ is a real-valued reward function.
- $\alpha$ is a finite set of agents.
- $A_i$ is the finite set of actions available to agent $i$.
- $\mathcal{T} : S \times A_1 \times \cdots \times A_n \times S \to [0, 1]$ is a transition function that determines the next state given all of the agents’ actions.

[^UtilityLearningHumans2019]: Carroll, M., Shah, R., Ho, M. K., Griffiths, T., Seshia, S., Abbeel, P., & Dragan, A. (2019). On the Utility of Learning about Humans for Human-AI Coordination. Advances in Neural Information Processing Systems, 32. https://proceedings.neurips.cc/paper_files/paper/2019/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html

---

## 历史遗留问题

早期的 Overcooked-AI 组件版本依赖复杂，且不向下兼容。

**截至 2023 年**，CHAI 团队已经将上述组件合并发布至 [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) 仓库。与 `neurips2019` 版本相比，当前版本的套件有大量优化，包括 `human_aware_rl` 引入 [Ray](https://docs.ray.io/en/latest/rllib/rllib-training.html) 作为分布式训练框架，`overcooked_ai_py` 在游戏中加入新动作 `煮食材`，`overcooked_demo` 可一键更新 `overcooked_ai_py` 版本并在 Web 演示游戏，以及更丰富的文档和用例。

然而，目前所调查的相关工作均使用 `neurips2019` [^UtilityLearningHumans2019] 版本实现自己的算法，且不被当前版本的套件兼容。

**因此，目前将以 `neurips2019` 版本为基础复现相关工作**。

[^UtilityLearningHumans2019]: Carroll, M., Shah, R., Ho, M. K., Griffiths, T., Seshia, S., Abbeel, P., & Dragan, A. (2019). On the Utility of Learning about Humans for Human-AI Coordination. Advances in Neural Information Processing Systems, 32. https://proceedings.neurips.cc/paper_files/paper/2019/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html

---

## 改进方向

后期，为方便 zero-shot coordination（ZSC）研究者享受现代 benchmark env 的特性，可以尝试：

1. 将 `neurips2019` 版的模型转换为兼容当前版本的格式，或
1. 使用当前版本 Overcooked-AI 或 Melting Pot [^MeltingPot2023] [^MeltingPotResearch2023] 复现相关工作

> Melting Pot 是 DeepMind 团队提出的更现代的 MARL benchmark 环境，其同样实现了 Overcooked 游戏，可被视为 Overcooked-AI 的超集。Melting Pot 聚焦于社会情境下的多智能体互动，具有更丰富的特性，例如：更精细的环境设置（可调的观测视窗）、更多的互动任务（游戏）、更多的玩家数量，以及更合理的评估指标，有望成为 ZSC 研究的新标杆。

**今后的有关 MARL 的工作可以考虑使用 Melting Pot 作为 simulater。**

[^MeltingPot2023]: Agapiou, J. P., Vezhnevets, A. S., Duéñez-Guzmán, E. A., Matyas, J., Mao, Y., Sunehag, P., Köster, R., Madhushani, U., Kopparapu, K., Comanescu, R., Strouse, D. J., Johanson, M. B., Singh, S., Haas, J., Mordatch, I., Mobbs, D., & Leibo, J. Z. (2023). Melting Pot 2.0 (arXiv:2211.13746). arXiv. https://doi.org/10.48550/arXiv.2211.13746

[^MeltingPotResearch2023]: Hu Y. (2023). Melting Pot Research Report. https://slidev.metattri.com/

---
layout: center
class: "text-center"
---

# Zero-Shot Coordination Baselines

> Zero-Shot Coordination：与没见过的伙伴（人或 AI）协作
> 
> 以下工作大部分使用 Overcooked-AI 评估算法性能

---

## On the Utility of Learning about Humans for Human-AI Coordination (HARL) [^UtilityLearningHumans2019]

2019 年，Self-play（SP）和 population-based training（PBT）是两种常用的 MARL 训练策略，用于训练与人类协作的 agents。

本文认为，SP 和 PBT agent 将假设其伙伴是最优的或者与自己相似的（而人类的行为不是最优的且难以预测），这会导致 agent 更适合跟自身而非人类协作，将人类数据或模型纳入训练过程将改进 human-AI coordination 的性能。因此，本文设计了 Overcooked-AI 环境，并提出了：

- Behavior Cloning model（BC）、 Proxy human model H$_{Proxy}$
  > 二者都是使用人类数据训练的动作分类（预测）器，BC 是参与训练 agent 的伙伴；而 H$_{Proxy}$ 作为 ground truth，用于评估 agent 的性能，二者关系类似于 训练集 和 测试集
- 2 类与人类协作的 agent 模型
  - 不使用人类数据：Self-Play（SP）、Population-Based Training（PBT）、规划方法
  - 使用人类数据：PPO with human model PPO$_{BC}$、规划方法

[^UtilityLearningHumans2019]: Carroll, M., Shah, R., Ho, M. K., Griffiths, T., Seshia, S., Abbeel, P., & Dragan, A. (2019). On the Utility of Learning about Humans for Human-AI Coordination. Advances in Neural Information Processing Systems, 32. https://proceedings.neurips.cc/paper_files/paper/2019/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html

---

### Method

**Self-Play**：paly with self in each iteration, using PPO.

**Population-Based Training（PBT）**：play with n agents in each iteration, using PPO

- 种群规模 $n=3$（本文中），每个 agent 与 SP agent 结构相同，只是伙伴从自己变成了不同的 agents
- PBT 算法可简述为：初始化 n agents，两两配对训练，最差的 agent 变异（进化），细节如下：
  
  ```python
  while not converged:
    for i in range(n):
      for j in range(i+1, n):
        train(agent[i]) # training agent_i using PPO and agent_j is embedded into the environment
      performance[i] = eval(agent[i])
    worst_agent = get_worst(performance)
    agent[worst_agent] = mutate(agent[worst_agent])
  ```

---

**PPO$_{BC}$**：play with BC in each iteration, using PPO

1. 使用人类游戏数据训练行为克隆（behavior cloning）模型 BC
   - 分类任务
   - 使用 cross-entropy loss
2. BC 作为环境的一部分，使用 PPO 作为策略梯度算法，训练agent

---
layout: figure
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ad7022db181.png
---

### Evaluation

#### AI-H$_{Proxy}$ Play

---
layout: figure
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ad6faa1e503.png
---

#### AI-Human Play

---

## Fictitious Co-Play (FCP) [^strouseCollaboratingHumansHuman2021]

**Motivation**：

1. Self Play（SP）或 Population Play（PP），产生的 agent 过度适应他们的训练伙伴，难以推广到人类
2. HARL 提出的 PPO$_{BC}$（本文称为 BCP） 涉及到收集大量人类数据，繁重而昂贵
3. 与 novel partner 的合作需要处理对称问题，比如：二人相遇时的避让策略，同左 or 同右？
4. 与人类合作需要迅速理解并适应他们的个人优势、劣势和偏好
5. 好的 agent 应该能够和各水平的伙伴合作，而不是只能和最优伙伴合作

**Contribution**：

- 提出 Fictitious Co-Play（FCP）来训练能够与人类进行 zero-shot 协调的 agent
- 证明 FCP agent 在与各种 agents 进行 zero-shot 协调时，比 SP、PP 和 BCP 的表现更好
- 证明 FCP 在任务得分和人类偏好方面都明显优于 BCP 的 SOTA

[^strouseCollaboratingHumansHuman2021]: Strouse, D., McKee, K., Botvinick, M., Hughes, E., & Everett, R. (2021). Collaborating with Humans without Human Data. Advances in Neural Information Processing Systems, 34, 14502–14515. https://proceedings.neurips.cc/paper/2021/hash/797134c3e42371bb4979a462eb2f042a-Abstract.html

---
layout: figure-side
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ad7277e4d7d.png
---

### Method

**Stage 1**：**独立**训练 n 个 SP agents，保存各阶段的 checkpoint 至 pool（代表不同水平）

**Stage 2**：与 pool 中 agents 配对训练 FCP agent

为了推广 FCP，使其能够接受视觉 observation，本文未采用 PPO，而是设计了强化学习算法：使用 V-MPO 算法，结合 ResNet 和 LSTM 构建所有 agent（stage 1 & 2），在分布式环境并行训练。

> “For our reinforcement learning agents, we use the V-MPO [65] algorithm along with a ResNet [26] plus LSTM [29] architecture which we found led to optimal behavior across all layouts. Agents are trained using a distributed set of environments running in parallel [17], each sampling two agents from the training population to play together every episode.” ([Strouse 等, 2021, p. 4](zotero://select/library/items/YRLFN64D)) ([pdf](zotero://open-pdf/library/items/U7HJDG94?page=4&annotation=RB979D7S))

---

### Evaluation

本文使用 3 类 agent，与 FCP 和 baselines 配对玩游戏，比较上菜次数（Deliveries）：

- Proxy human H$_{Proxy}$
- SP agent（as skillfull partner）
- 随机初始化的策略 agent（as low-skill partner）

![image-20240122230242572](https://lsky.metattri.com/i/2024/01/22/65ae839583503.png)

---

#### 消融实验

- FCP: pool 中 agents 结构相同，seed 不同，Stage 2 使用过去 checkpoints。
- FCP$_{-T}$: 相比于 FCP，不使用过去 checkpoints（未收敛的 agents），用于测试过程中生成的 checkpoints 的重要性。
- FCP$_{+A}$: 相比于 FCP，agents 结构不同，用于测试不同的结构会不会带来更好的多样性。
- FCP$_{-T,+A}$: 相比于 FCP$_{+A}$，不使用过去 checkpoints，用于测试不同结构能否替代过程中生成的 checkpoints。

![image-20240122231418688](https://lsky.metattri.com/i/2024/01/22/65ae864c85547.png)

---

**然而**：

- FCP 不仅耗费时间，而且容易出现研究者的偏见，可能会对创建的 agent 的行为产生负面影响。

- 对于更加复杂的游戏，FCP 可能需要更大的 pool，这可能是不切实际的。

---

## Trajectory Diversity (TrajeDi) [^lupuTrajectoryDiversityZeroShot2021]

TBD

MEP 传承了 TrajeDi 的思想，并达到新 SOTA。

[^lupuTrajectoryDiversityZeroShot2021]: Lupu, A., Cui, B., Hu, H., & Foerster, J. (2021). Trajectory Diversity for Zero-Shot Coordination. Proceedings of the 38th International Conference on Machine Learning, 7204–7213. https://proceedings.mlr.press/v139/lupu21a.html

---

## Maximum Entropy PBT (MEP) [^zhaoMaximumEntropyPopulationBased2023]

### TL;DR

竞争环境下，SP 和 PBT 效果较好，但在与人类合作的环境下，二者会训练出过于 specific 的策略。

一种解决思路是，引入人类数据辅助训练，但数据收集成本较高；

另一种思路是提高参与训练的 agents 的多样性：

- **diverse set of policies**：例如 TrajeDi 优化 trajectory 间的 JS 散度从而达到 diverse 的目标，FCP 则使用随机种子或不同的 checkpoints；
- **domain randomization**：some features of the environment are changed randomly during training to make the policy robust to that feature，本文的方法可被视为 domain randomization。同时，本文采用最大熵强化学习（MERL），相比于一般的强化学习，MERL 则需要最大化 return + 熵，这样会使得策略更具有**探索性**并且具有更强的**鲁棒性**。

[^zhaoMaximumEntropyPopulationBased2023]: Zhao, R., Song, J., Yuan, Y., Hu, H., Gao, Y., Wu, Y., Sun, Z., & Yang, W. (2023). Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination. Proceedings of the AAAI Conference on Artificial Intelligence, 37, 6145–6153. https://doi.org/10.1609/aaai.v37i5.25758

---

### Method

与 FCP 类似，MEP 也是两阶段法：首先训练一个 maximum entropy population，然后通过 population 训练一个 robust agent。

本文借鉴最大熵强化学习的思想修改了训练的目标函数，涉及两个概念：**Population Diversity & Entropy**：

**Population Diversity**：首先需要 population 中 agents 自身的策略更有探索性，同时也需要两两 agents 的策略差异更大。

$$
\mathrm{PD}\left(\left\{\pi^{(1)}, \pi^{(2)}, \ldots, \pi^{(n)}\right\}, s_{t}\right):=\frac{1}{n} \sum_{i=1}^{n} \mathcal{H}\left(\pi^{(i)}\left(\cdot \mid s_{t}\right)\right)
+\frac{1}{n^{2}} \sum_{i=1}^{n} \sum_{j=1}^{n} D_{\mathrm{KL}}\left(\pi^{(i)}\left(\cdot \mid s_{t}\right), \pi^{(j)}\left(\cdot \mid s_{t}\right)\right)
$$

where KL-divergence ($D_{\mathrm{KL}}$) and entropy ($\mathcal{H}$) are defined as follows:

$$
D_{\mathrm{KL}}\left(\pi^{(i)}\left(\cdot \mid s_{t}\right), \pi^{(j)}\left(\cdot \mid s_{t}\right)\right)=
\sum_{a \in \mathcal{A}} \pi^{(i)}\left(a_{t} \mid s_{t}\right) \log \frac{\pi^{(i)}\left(a_{t} \mid s_{t}\right)}{\pi^{(j)}\left(a_{t} \mid s_{t}\right)}
$$

$$
\mathcal{H}\left(\pi^{(i)}\left(\cdot \mid s_{t}\right)\right)=-\sum_{a \in \mathcal{A}} \pi^{(i)}\left(a_{t} \mid s_{t}\right) \log \pi^{(i)}\left(a_{t} \mid s_{t}\right)
$$

---

**Population Entropy**：因为 PD 计算复杂度过高，并且 KL 散度是 unbounded 的，可能会有收敛性的问题，因此本文提出了 PE（population mean policy 的熵），其具有线性复杂度并且是 bounded 的，作为 PD 的 surrogate loss。文中也证明了 PE 是 PD 的 lower bound，因此可以作为 surrogate loss。

$$
\mathrm{PE}\left(\left\{\pi^{(1)}, \pi^{(2)}, \ldots, \pi^{(n)}\right\}, s_{t}\right): = \mathcal{H}\left(\bar{\pi}\left(\cdot \mid s_{t}\right)\right),
\text { where } \bar{\pi}\left(a_{t} \mid s_{t}\right): = \frac{1}{n} \sum_{i = 1}^{n} \pi^{(i)}\left(a_{t} \mid s_{t}\right)
$$

为了训练出能 coorperate well 又 mutually distinct 的 strategy，本文在目标函数中引入 PE 分量，同时也引入了 hyperparameter $\alpha$ 来控制 PE 的权重，作为 **MEP training objective**：

$$
J(\bar{\pi})=\sum_t\mathbb{E}_{(s_t,a_t)\sim\bar{\pi}}\left[R(s_t,a_t)+\alpha\mathcal{H}(\bar{\pi}(\cdot|s_t))\right]
$$

---

#### **Stage 1**: train a maximum entropy population

1. 随机从 population 中采一个 agent
2. 然后优化该 agent 的策略
3. 重复步骤 1 - 2，直到 $J(\bar{\pi})$ 收敛。

![image-20240122172348406](https://lsky.metattri.com/i/2024/01/22/65ae342bd4d9e.png)

> $r(s_t, a_t)$ 的获取是由采得的 agent 以及他的 copy 作为 partner 得到的，相当于 SP

---

#### **Stage 2**: Training a robust agent (MEP Agent) paired with MEPooulation

本文没有直接对 MEpopulation 做 uniformly sample 来获得伙伴 agent 与 MEP agent 配对训练，而是使用了 learning progress-based prioritized sampling（LPPS）来选择伙伴。LPPS 会选择 learning progress 最大的伙伴，这样可以使得 MEP agent 更具有探索性。

对于具体的 LPPS 方法，本文未采用 maximize average（最大化对 population 中所有 partner 的表现的平均值）， 因为 MEP agent 可能会学到与最容易合作的伙伴合作的策略，而放弃了难以合作的。因本文用 ranked-based 优先级采样让 MEP agent 优先跟难以合作的伙伴配对训练：

$$
p(\pi^{(i)})=\frac{\operatorname{rank}\left(1/\mathbb{E}_\tau\left[\sum_tR(s_t,a_t^{(A)},a_t^{(i)})\right]\right)^\beta}{\sum_{j=1}^n\operatorname{rank}\left(1/\mathbb{E}_\tau\left[\sum_tR(s_t,a_t^{(A)},a_t^{(j)})\right]\right)^\beta}
$$

优先级采样是 smooth approximation of maximize minimum（极端情况下只和最难合作的进行训练就是 maximize minimum 了），当 population 足够多时，会有 partner agent 的策略与人类策略 ε-close，文中也证明了 human-ai coordination 的一些下界的性质。

---
layout: figure
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ae7af17a694.png
---

### Evaluation

#### AI-H$_{Proxy}$ Play

---

## Hidden-Utility Self-Play (HSP) [^yuLearningZeroShotCooperation]

TBD

[^yuLearningZeroShotCooperation]: Yu, C., Gao, J., Liu, W., Xu, B., Tang, H., Yang, J., Wang, Y., & Wu, Y. (n.d.). Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased.

---

## PECAN [^louPECANLeveragingPolicy2023]

Policy Ensemble Context-Aware zero-shot human-AI coordinatioN

<!-- ### Method -->
---
layout: figure
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ad721756cbf.png
---

<!-- ![image-20240122033547773](https://lsky.metattri.com/i/2024/01/22/65ad721756cbf.png) -->

[^louPECANLeveragingPolicy2023]: Lou, X., Guo, J., Zhang, J., Wang, J., Huang, K., & Du, Y. (2023). PECAN: Leveraging Policy Ensemble for Context-Aware Zero-Shot Human-AI Coordination. Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, 679–688.

---

## Cooperative Open-ended LEarning (COLE) [^CooperativeOpenendedLearning2023]

### Method

![image-20240123004328668](https://lsky.metattri.com/i/2024/01/23/65ae9b324dde7.png)

[^CooperativeOpenendedLearning2023]: Li, Y., Zhang, S., Sun, J., Du, Y., Wen, Y., Wang, X., & Pan, W. (2023). Cooperative Open-ended Learning Framework for Zero-Shot Coordination. Proceedings of the 40th International Conference on Machine Learning, 20470–20484. https://proceedings.mlr.press/v202/li23au.html

---

### Evaluation

#### AI-H$_{Proxy}$ Play

---
layout: figure
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ae8b17798cb.png
---

<!-- ![image-20240122233445010](https://lsky.metattri.com/i/2024/01/22/65ae8b17798cb.png) -->

---
layout: figure
figureUrl: https://lsky.metattri.com/i/2024/01/22/65ae8b2899e17.png
---

#### AI-AI Play

---
layout: end
hideInToc: true
---

# Thank you!

> 胡逸同，2024/01/25
