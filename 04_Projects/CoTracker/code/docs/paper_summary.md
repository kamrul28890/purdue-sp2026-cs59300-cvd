# CoTracker3 Paper Summary

## One-paragraph summary

CoTracker3 is a point-tracking model for videos. Its job is to follow selected points from frame to frame, even when the scene moves, the object deforms, or parts of it become occluded. The paper argues that older trackers rely too much on synthetic data because real point-tracking labels are hard to annotate, and that this creates a gap between training and real-world performance. Their answer is a simpler tracking model plus a training recipe that creates pseudo-labels on real unlabeled videos using teacher models. The result is a tracker that is simpler, uses far less data than previous top systems, and performs very strongly on standard point-tracking benchmarks.

## What problem is the paper trying to solve?

The paper is trying to solve this core problem:

- how do we train strong point trackers for real videos when dense point-tracking labels are expensive and difficult to obtain?

The authors say that many prior trackers are trained mostly on synthetic data, which helps with supervision but creates a synthetic-to-real gap.

## What did they do?

They did two main things:

1. Built a new tracker called **CoTracker3**
   - simpler than earlier trackers
   - available in **online** and **offline** versions
   - designed to track points reliably through visible and occluded periods

2. Proposed a simpler semi-supervised training recipe
   - use off-the-shelf teacher models to create pseudo-labels on real videos
   - combine that with synthetic training data
   - study how performance changes as more real unlabeled data are added

## What was the goal?

The goal was not just to make a tracker that works.

The goal was to make a tracker that is:

- strong on real videos
- simpler than previous approaches
- easier to train
- more data-efficient

## What were the results?

From the official CoTracker repository README, CoTracker3 improves over earlier CoTracker variants on TAP-Vid benchmarks.

Reported README numbers for CoTracker3:

- **Offline**
  - Kinetics: `67.8`
  - DAVIS: `76.9`
  - RoboTAP: `78.0`
  - RGB-Stacking: `85.0`
- **Online**
  - Kinetics: `68.3`
  - DAVIS: `76.7`
  - RoboTAP: `78.8`
  - RGB-Stacking: `82.7`

The paper abstract also emphasizes that the method achieves better results while using **1,000x less data** than previous top-performing approaches.

## What is the output product of the model?

The model takes a video and a set of query points, or a grid of sampled points, and returns:

- point trajectories across frames
- visibility / occlusion predictions for those points

In practical terms, the output is:

- where each tracked point moves in each frame
- whether the model thinks that point is still visible

This is why the examples in the repo return `pred_tracks` and `pred_visibility`.

## What does it do in plain language?

In plain language, CoTracker3 answers:

- "If I pick this point here, where does it go over time?"

And it can do that for many points jointly, not just one.

That makes it useful for:

- motion understanding
- object tracking
- video segmentation support
- structure-from-motion style pipelines
- general spatiotemporal analysis

## Why is this a good class project?

It is a good project because it gives us all three things we need:

- a modern paper with a strong technical story
- a working open-source codebase
- both quantitative and qualitative evaluation paths

For our project specifically, the best angle is:

- reproduce the baseline
- compare online vs offline behavior
- analyze failure modes on custom videos
