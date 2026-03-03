# Recommendation Systems with Reinforcement Learning

Every time you open Netflix, the app shows you a wall of movies. Out of thousands of options, it picks the ones it thinks you will like. But here is the strange part — it does not actually know what you like. It has to guess, watch your reaction, and learn. How does it get better at guessing over time?

---

**Before you start, you need to know:**
- What reinforcement learning does at a high level (an agent takes actions, gets rewards, and learns) — covered in `../fundamentals/what-is-reinforcement-learning.md`
- What exploration vs exploitation means informally (trying new things vs sticking with what works) — covered in `../fundamentals/what-is-reinforcement-learning.md`

---

## The analogy: a DJ at a party

Imagine you are a DJ at a party where you do not know anyone. Your job is to keep as many people dancing as possible. You start with no idea what songs they like.

You play a pop song. Half the crowd dances. Good — but not great. Now you have a choice. Do you keep playing pop (it worked okay) or do you try jazz (it might be amazing, or it might empty the floor)?

If you only play pop because it worked once, you might never discover that this crowd loves jazz. But if you keep trying random genres all night, you waste time on songs nobody likes.

The best DJs do both. They mostly play what is working, but every few songs they try something new. Over the night, they learn the crowd's taste and the dance floor stays full.

### What the analogy gets right

- The DJ (recommendation system) does not know the crowd's taste in advance — it must learn from reactions
- Playing a song is like recommending a product — it is a choice with an uncertain outcome
- The crowd dancing is like a user clicking — it is the reward signal
- The explore-exploit balance is the core challenge — play it safe or try something new?
- Over time, the DJ builds a model of what works, just like the algorithm builds estimated values for each option

### The concept in plain words

Recommendation systems use RL to solve the explore-exploit problem. There are three levels of sophistication.

**Multi-armed bandits** are the simplest version. Imagine a row of slot machines. Each machine pays out at a different rate, but you do not know which rates are which. You pull one machine, see what happens, and decide which to try next. In recommendations, each "machine" is a product you could show the user. Each pull is one recommendation. The payout is whether the user clicks.

Three strategies handle this:
- **Epsilon-greedy** mostly picks the best option so far, but with a small chance (say 10%) picks a random one instead. Simple but effective.
- **UCB (Upper Confidence Bound)** gives a bonus to options that have not been tried much. The less you know about an option, the more optimistic you are about it. This naturally balances exploring unknown options and exploiting known good ones.
- **Thompson Sampling** keeps a belief about how good each option is. Instead of picking the best estimate, it samples from each belief and picks the highest sample. Options you are uncertain about sometimes sample high, so they get tried. Options you are confident about consistently sample high, so they get picked often.

**Contextual bandits** add personalization. Plain bandits recommend the same thing to everyone. But a 25-year-old action fan and a 60-year-old documentary fan want different movies. Contextual bandits look at who the user is (their context) before choosing what to recommend. The algorithm learns: "for users like this, product X works well."

**Full RL** considers long-term effects. Bandits treat each recommendation as independent. But in reality, showing someone three action movies in a row makes them bored of action. Full RL models how the user's state changes over time — preferences shift, boredom builds, novelty fades. It optimizes for long-term engagement, not just the next click.

### Where the analogy breaks down

A real DJ reads body language, facial expressions, and energy levels all at once. A recommendation system only sees clicks, watch time, and skips — much simpler signals. Also, a DJ handles one crowd, but a recommendation system handles millions of different users simultaneously, each with their own taste.

---

**Quick check — can you answer these?**
- Why can a recommendation system not just always show the most popular item?
- What is the difference between epsilon-greedy and UCB?
- Why do full RL recommendations outperform bandits when user preferences change over time?

If you cannot answer one, re-read that part. That is completely normal.

---

You just learned how RL powers the recommendation systems behind Netflix, Spotify, YouTube, and Amazon. The same explore-exploit ideas from multi-armed bandits scale all the way up to deep RL systems that personalize content for billions of users. Every time an app shows you something you love that you never would have searched for — that is RL doing its job.

**Ready to go deeper?** Head to [recommendation-systems-interview.md](./recommendation-systems-interview.md) for the full math, regret analysis, and interview-grade depth.
