---
layout: default
title: proposal
---

## Summary of the Project
Our project aims to explore and compare the efficacy of various reinforcement learning (RL) algorithms in playing the game 2048, which is a puzzle game on a 4x4 grid where tiles of equal numbers are merged to create larger tiles until the 2048 tile is reached. Because there have been a variety of previous attempts across different contexts showing the surprising difficulty of this task, our focus is to (1) find and optimize an algorithm that can learn to reach the 2048 tile, and (2) perform a comparative analysis of Policy Gradient Algorithms and Monte Carlo Tree Search (MCTS) for this game under a single, controlled setting. We will use a Python implementation of the 2048 game so that we can use a 2D array game state as input, and the output will always be one of four possible moves: up, down, left, or right. By conducting this experiment, we hope to discover the strengths and weaknesses of each approach which can inform future studies regarding RL for solving complex puzzles.

## AI/ML Algorithms
We anticipate performing a comparison analysis of Policy Gradient Algorithms (Proximal Policy Optimization, Advantage Actor-Critic, etc.) and Monte Carlo Tree Search (MCTS).

## Evaluation Plan
To evaluate our project's success, we will use quantitative metrics including the average maximum tile reached, average score achieved across a set of games, and the distribution of maximum tiles reached by each reinforcement learning algorithm. As baselines, we will compare the algorithms' performance against a model that performs random moves and one that only prioritizes merging the largest tiles. We expect our RL approaches to significantly outperform these baselines by at least doubling their average maximum scores. This evaluation will be conducted on a large set of games to ensure that the results are statistically significant.

Additionally, we will perform qualitative analysis by testing the algorithms on a smaller, 3x3 grid to verify that they work as expected before testing them on the real game. We will use visuals such as heat maps to see how action probabilities are changing across the grid as the game progresses, and watch the models "play" the game step by step to analyze their learning. Our moonshot case would be surpassing not only 2048, but reaching the next tile, 4096, which has been demonstrated by other attempts to be very difficult. This would be a major success and be greatly beneficial to other RL game studies. 

## Meet the Instructor
We met with Professor Roy Fox on 1/17/25 to discuss our initial ideas and plan. Through this meeting, we learned about using Monte Carlo Tree Search (MCTS) and received advice about prioritizing "order" in the game state over immediately creating larger tiles.

## AI Tool Usage
We did not use any AI tools to assist in writing this proposal or any code that was a part of this project. This section will be updated if we decide to utilize any AI tools at a later stage.