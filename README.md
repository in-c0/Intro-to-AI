# Introduction to Artificial Intelligence

Self-study notes for a solid foundation in AI, inspired by UNSW COMP9814 Extended Artificial Intelligence (24T3).

**Read it here → https://in-c0.github.io/Intro-to-AI/**

| | Chapter |
|---|---|
| 1 | [Introduction](docs/01-introduction.md) — history of AI, agents, knowledge representation |
| 2 | [Search](docs/02-search.md) — BFS, DFS, iterative deepening, informed search |
| 3 | [Artificial Neural Networks](docs/03-neural-networks.md) — perceptrons, MLPs, network design |
| 4 | [Reinforcement Learning](docs/04-reinforcement-learning.md) — action values, returns, value functions, MC & TD |
| 5 | [Metaheuristics](docs/05-metaheuristics.md) — gradient methods, simulated annealing, tabu search, GAs |

Projects: [Mountain Car with Gym](docs/gym.md) · [Feedforward Neural Network](docs/Feedforward-Neural-Network.md)

## Build locally

```bash
pip install -r requirements.txt
mkdocs serve        # http://127.0.0.1:8000/Intro-to-AI/
mkdocs build --strict   # same check CI runs; fails on broken links/anchors
```

## Contributing

Fixes and new sections are welcome — open an issue or a pull request. Every PR is link-checked automatically.

## License

[MIT](LICENSE)
