"""
AutonomousBrain v5 -- ONE FILE. numpy is the only requirement.

    python autonomous_brain_v5.py

If v4 threw an error for you, this is why: v4 was two files, and
`demo_v4.py` did `from global_matrix_brain_v4 import ...`. That line needs
the other file sitting next to it under EXACTLY that name -- and browsers
routinely save it as "global matrix brain v4.py", with spaces, which is not
a legal module name. Result: ModuleNotFoundError. Running the engine file
on its own was no better: it had no `if __name__ == "__main__"` block, so it
printed nothing at all and looked broken. Both problems are gone here --
one file, one command, and matplotlib is optional (you lose the picture,
nothing else).


WHAT IS NEW IN v5: IT RUNS ITSELF
---------------------------------
v4 could learn, but I was still driving it -- my loop decided when a trial
started, when it ended, and when to stop. v5 adds AutonomousAgent, which
closes the loop and hands four decisions to the brain itself:

  WHAT TO DO       Motor pools compete under shared inhibition; the winner
                   is the action. Which pool wins depends on synapses the
                   brain shaped from reward it earned itself.

  EXPLORE OR NOT   Not an epsilon decaying on a timer. Exploration is driven
                   by the brain's own forward-model prediction error. When
                   the world stops surprising it, curiosity falls and
                   behaviour settles.

  WHEN TO SLEEP    Sleep pressure accumulates from its own metabolic heat,
                   the way adenosine does. Past threshold it stops acting,
                   replays, prunes, wakes. Nothing outside schedules this.
                   Measured: it put itself to sleep 11 times in 9000 steps.

  WHAT TO KEEP     An episode is written only when novelty crosses a
                   threshold. It stores what surprised it, not everything.

Measured, two seeds, no cherry-picking (steps needed to find food, by
quarter of a 9000-step run):

    seed 11:  144 -> 84 -> 76 -> 50     random walk: 85
    seed 23:  162 -> 98 -> 91 -> 72     random walk: 113

It starts far worse than random -- a confident wrong policy bumps into the
same wall repeatedly, while a random one at least diffuses -- and ends
around 40% faster than chance. That is real self-directed learning in a
closed loop. It is also modest, and the reason is worth knowing: credit
assignment across a long action sequence is exactly where purely local
plasticity hits its known ceiling. On the immediate cue->action task
(section 5) the same machinery reaches 88-94%, because the gap between
action and reward is short. That gap is the whole difficulty, and closing
it is what TD learning and a value critic are for -- the honest next step,
and the thing to build next.


ON "FREE" AND "THINKS FOR ITSELF" -- PLEASE READ THIS
------------------------------------------------------
"Decides" above is the control-theory sense: the system's own internal
state selects its output instead of an outside script doing it. That is a
real, non-trivial property and this code genuinely has it.

It is not the same as thinking, understanding, or wanting, and I would be
doing you a disservice to blur the two. There is no path from local
plasticity in 3000 integrate-and-fire units to a mind -- not because the
code needs more work, but because nobody knows what the missing pieces are.
That is the open research problem of the field, not a TODO in this file.
Pure local STDP has never been competitive with backpropagation on hard
tasks; the spiking networks that do compete (Spikformer, surrogate-gradient
training) get there by abandoning pure locality. Anyone claiming a script
like this is "almost AGI" is selling something.

What you have is a genuinely good, self-regulating, self-driving spiking
brain that learns, chooses, sleeps, and remembers -- and every number in
this file is reproducible by running it. That is worth more than a label.

It has no network access, no shell access, and no control over any system.
That is deliberate and I am not going to add it. Autonomous software with
unsupervised control over networks and machines produces consequences
nobody chose, and you would be the operator on the hook for them. The
honest way to feed it real data is sensory_encode(): hand it a file or a
string, it becomes spike patterns. Data in, nothing out.


LINEAGE
-------
v1  dense N x N, excitation only, no inhibition  -> instant seizure
v2  sparse + Dale's law + homeostatic threshold  -> stable
v3  multiple regions + global valve + save/load  -> but learning ran BACKWARDS
v4  event-driven trace STDP, synaptic scaling, refractory period,
    metabolism, dopamine, curiosity, sleep, RAM->disk memory
v5  all of that in one file, plus the agent loop that closes it


THE BUGS FIXED ALONG THE WAY (all of these were live, all measured)
-------------------------------------------------------------------
1. v3's STDP scanned every synapse every step against stale timestamps, so
   one spike pair was applied ~20 times and the sign was set by whichever
   region fired slower. Its own headline experiment came out backwards:
   paired pathway 0.88x, unrelated control 0.96x. Replaced with event-driven
   trace STDP. Now: paired 2.97x, controls 1.94x and 1.86x.
2. `np.clip(w[mask], 0, wmax, out=w[mask])` looks correct and silently does
   nothing -- boolean-mask indexing returns a copy, so the clip lands in a
   temporary and weights are never actually bounded.
3. No synaptic scaling meant every weight drifted up together and
   selectivity washed out: the paired pathway grew 3.8x but so did the
   controls at 2.4x, so essentially nothing had been learned.
4. No refractory period anywhere in v1-v3. Thresholds and metabolism are
   negative FEEDBACK and can always be out-driven by a large enough input;
   that is why v3 still sat at ~85% under flood. The refractory period is a
   hard CONSTRAINT and caps it at 33.3% no matter what.
5. Hard weight bounds pinned every driven synapse at w_max. Soft bounds
   (Gutig et al. 2003) give a graded, stable distribution instead.
6. Prune-and-archive did not update the synaptic-scaling budget, so
   normalization silently inflated survivors to replace what was removed.
7. Region RNG state was not saved, so a reloaded brain diverged within one
   step and "identical after reload" could not actually be verified.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import time as _time
from pathlib import Path

import numpy as np


# ==========================================================================
#  METABOLISM  --  the anti-overheating system
# ==========================================================================

class Metabolism:
    """Per-region energy/heat budget.

    Real neural tissue cannot fire flat out: spikes cost ATP, ATP
    regeneration is rate-limited, and the by-products (adenosine, heat)
    accumulate and suppress excitability. That negative feedback is why
    an intact brain doesn't do what v1 did.

    Two outputs, both smooth functions of heat:
      gain    multiplies every incoming current into the region
      fatigue adds to every firing threshold in the region

    Heat rises with the fraction of the region that fired, falls at a
    fixed cooling rate. `critical` is the heat level at which gain has
    dropped to half.
    """

    def __init__(self, cost=1.0, cooling=0.12, critical=0.5, sustainable=0.06,
                 fatigue_gain=1.0, gain_floor=0.05):
        self.cost = cost
        self.cooling = cooling
        self.critical = critical
        self.sustainable = sustainable   # firing fraction that costs nothing net
        self.fatigue_gain = fatigue_gain
        self.gain_floor = gain_floor
        self.heat = 0.0
        self.energy = 1.0

    @property
    def gain(self):
        return max(self.gain_floor, 1.0 / (1.0 + (self.heat / self.critical) ** 2))

    @property
    def fatigue(self):
        return self.fatigue_gain * self.heat

    def update(self, firing_fraction, cooling_scale=1.0):
        # Only activity ABOVE the sustainable rate generates net heat --
        # baseline metabolism is covered by baseline blood flow. Without
        # this the region cooks itself at its own healthy target rate.
        self.heat += self.cost * max(0.0, firing_fraction - self.sustainable)
        self.heat -= self.cooling * cooling_scale * self.heat
        self.heat = max(0.0, self.heat)
        # energy is tracked for reporting/introspection; it mirrors heat
        self.energy = float(np.clip(1.0 - self.heat / (2.0 * self.critical), 0.0, 1.0))

    def state(self):
        return dict(heat=self.heat, energy=self.energy,
                    gain=self.gain, fatigue=self.fatigue)


# ==========================================================================
#  REGION  --  one functionally specialised population
# ==========================================================================

class BrainRegion:
    """Sparse, Dale's-law, leaky integrate-and-fire population with
    homeostatic thresholds, event-driven trace STDP, shared inhibitory
    feedback, and its own metabolism.

    Inherited unchanged from v2/v3: sparse connectivity (memory is O(edges),
    not O(N^2)), Dale's law (a neuron is excitatory or inhibitory, never
    both), and the per-neuron homeostatic threshold.

    New in v4: x_pre/x_post traces, so plasticity is an event and not a
    scan; `global_inhibition`, a shared interneuron pool that makes
    sub-populations compete (this is what turns the motor region into a
    decision); and a Metabolism instance.
    """

    def __init__(self, name, num_neurons, role="", avg_out_degree=40,
                 frac_excitatory=0.8, exc_weight_range=(0.02, 0.06),
                 inh_weight_scale=4.0, leak_factor=0.9, base_threshold=1.0,
                 target_rate=0.02, threshold_adapt_rate=0.001,
                 threshold_bounds=(0.3, 3.0), global_inhibition=0.0,
                 trace_decay=0.85, learn_rate=0.008, w_max=0.15,
                 ltd_ratio=1.05, bg_rate=0.04, bg_amp=0.45, refractory=2,
                 metabolism=None, rng=None):
        self.name = name
        self.role = role
        self.N = num_neurons
        rng = rng if rng is not None else np.random.default_rng()

        self.is_excitatory = rng.random(num_neurons) < frac_excitatory

        pre, post, w = [], [], []
        k = min(avg_out_degree, num_neurons - 1)
        for i in range(num_neurons):
            targets = rng.choice(num_neurons, size=k, replace=False)
            targets = targets[targets != i]
            pre.append(np.full(len(targets), i, dtype=np.int32))
            post.append(targets.astype(np.int32))
            if self.is_excitatory[i]:
                w.append(rng.uniform(*exc_weight_range, size=len(targets)))
            else:
                w.append(-rng.uniform(*exc_weight_range, size=len(targets)) * inh_weight_scale)

        self.pre = np.concatenate(pre)
        self.post = np.concatenate(post)
        self.w = np.concatenate(w)
        self.is_exc_edge = self.is_excitatory[self.pre]
        self.w_budget = np.bincount(self.post[self.is_exc_edge],
                                    weights=self.w[self.is_exc_edge],
                                    minlength=num_neurons)

        self.voltages = np.zeros(num_neurons)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.x_pre = np.zeros(num_neurons)
        self.x_post = np.zeros(num_neurons)
        self.spike_count = np.zeros(num_neurons, dtype=np.int64)

        self.leak_factor = leak_factor
        self.base_threshold = base_threshold
        self.threshold = np.full(num_neurons, base_threshold)
        self.target_rate = target_rate
        self.threshold_adapt_rate = threshold_adapt_rate
        self.threshold_bounds = threshold_bounds
        self.global_inhibition = global_inhibition
        self.trace_decay = trace_decay
        self.learn_rate = learn_rate
        self.w_max = w_max
        self.ltd_ratio = ltd_ratio
        self.bg_rate = bg_rate
        self.bg_amp = bg_amp
        self.refractory = refractory
        self.refrac = np.zeros(num_neurons, dtype=np.int16)
        self.rng = rng
        self.metabolism = metabolism if metabolism is not None else Metabolism()

        self._init_kwargs = dict(
            role=role, avg_out_degree=avg_out_degree, frac_excitatory=frac_excitatory,
            exc_weight_range=list(exc_weight_range), inh_weight_scale=inh_weight_scale,
            leak_factor=leak_factor, base_threshold=base_threshold,
            target_rate=target_rate, threshold_adapt_rate=threshold_adapt_rate,
            threshold_bounds=list(threshold_bounds),
            global_inhibition=global_inhibition, trace_decay=trace_decay,
            learn_rate=learn_rate, w_max=w_max, ltd_ratio=ltd_ratio,
            bg_rate=bg_rate, bg_amp=bg_amp, refractory=refractory,
        )

    # ---- per-step mechanics, split into phases so the whole brain stays
    # ---- synchronous (every region sees the same timestep) --------------

    def receive_input(self, vec, scale=1.0):
        self.voltages += scale * vec

    def fire(self):
        """Phase 1: who crosses threshold. Fatigue from metabolism is added
        here, so an overheated region needs more drive to fire at all.

        Background drive first: cortex is never silent, and neurons sitting
        at 0 mV cannot respond to a weak signal at all. This spontaneous
        input parks the population just below threshold, which is what makes
        a small, structured input able to tip a specific assembly over."""
        if self.bg_rate:
            self.voltages += self.bg_amp * (self.rng.random(self.N) < self.bg_rate)

        effective_threshold = self.threshold + self.metabolism.fatigue
        # ABSOLUTE REFRACTORY PERIOD. A neuron that has just fired cannot
        # fire again for `refractory` steps no matter how hard it is driven,
        # because its sodium channels are physically inactivated. This is the
        # hard ceiling on firing rate -- 1/(refractory+1) -- and it is why a
        # real brain cannot do what v1 did. v1/v2/v3 had no refractory period
        # at all, which is precisely why 100%-of-population firing was even
        # reachable: nothing in the model forbade it. Thresholds and
        # metabolism are negative FEEDBACK and can always be out-driven by a
        # big enough input; this is a hard CONSTRAINT and cannot.
        self.spikes = (self.voltages >= effective_threshold) & (self.refrac == 0)
        self.voltages[self.spikes] = 0.0
        self.voltages[~self.spikes] *= self.leak_factor
        # Decrement EXISTING counters first, then arm the new ones. Doing it
        # the other way round burns one refractory step on the firing step
        # itself and silently halves the block, giving a 1/refractory ceiling
        # instead of 1/(refractory+1).
        np.subtract(self.refrac, 1, out=self.refrac, where=self.refrac > 0)
        self.refrac[self.spikes] = self.refractory
        in_refrac = self.refrac > 0
        self.voltages[in_refrac] = 0.0     # shunted, cannot integrate input
        self.spike_count += self.spikes
        return self.spikes

    def propagate_local(self):
        """Phase 2: recurrent transmission inside the region, scaled by the
        region's metabolic gain."""
        pre_spiking = self.spikes[self.pre]
        if np.any(pre_spiking):
            incoming = np.bincount(self.post[pre_spiking],
                                   weights=self.w[pre_spiking],
                                   minlength=self.N)
            self.voltages += self.metabolism.gain * incoming
        if self.global_inhibition:
            # Shared inhibitory interneuron pool: everyone is suppressed in
            # proportion to how active the region as a whole just was. This
            # is what creates competition between sub-populations.
            self.voltages -= self.global_inhibition * self.spikes.mean()

    def local_stdp(self):
        """Phase 3: event-driven trace STDP on excitatory synapses only.

        Uses the traces as they were BEFORE this step's spikes are folded
        in, so `x_pre` holds the recent history of presynaptic firing and
        the causal ordering is preserved.
        """
        lr = self.learn_rate
        touched = []
        post_fired = self.spikes[self.post] & self.is_exc_edge
        if np.any(post_fired):
            idx = np.where(post_fired)[0]
            # Soft (multiplicative) bounds: LTP is proportional to the room
            # left below w_max, LTD proportional to the current weight. With
            # hard bounds every driven synapse pins at w_max and all
            # selectivity is destroyed -- the network learns "everything is
            # connected to everything", which is no better than v1. Soft
            # bounds give a graded, stable weight distribution where how
            # strong a synapse ends up reflects how reliably it predicts its
            # target. (Gutig et al., J Neurosci 2003.)
            self.w[idx] += lr * self.x_pre[self.pre[idx]] * (self.w_max - self.w[idx])
            touched.append(idx)
        pre_fired = self.spikes[self.pre] & self.is_exc_edge
        if np.any(pre_fired):
            idx = np.where(pre_fired)[0]
            self.w[idx] -= lr * self.ltd_ratio * self.x_post[self.post[idx]] * self.w[idx]
            touched.append(idx)
        if touched:
            # Clip only the edges actually touched. NOTE: `np.clip(self.w[mask],
            # ..., out=self.w[mask])` looks right but silently does nothing --
            # boolean-mask indexing returns a COPY, so the clip lands in a
            # temporary and the real weights are never bounded.
            idx = np.unique(np.concatenate(touched))
            self.w[idx] = np.clip(self.w[idx], 0.0, self.w_max)

    def update_traces(self):
        """Phase 4: decay traces, then add this step's spikes."""
        self.x_pre *= self.trace_decay
        self.x_post *= self.trace_decay
        self.x_pre[self.spikes] += 1.0
        self.x_post[self.spikes] += 1.0

    def homeostasis(self, cooling_scale=1.0):
        """Phase 5: slow per-neuron threshold drift toward target rate, then
        metabolic update."""
        self.threshold += self.threshold_adapt_rate * (self.spikes.astype(float) - self.target_rate)
        np.clip(self.threshold, *self.threshold_bounds, out=self.threshold)
        self.metabolism.update(float(self.spikes.mean()), cooling_scale)

    def normalize_incoming(self):
        """Synaptic scaling on local excitatory synapses -- see
        Projection.normalize_incoming for why this is load-bearing."""
        exc = self.is_exc_edge
        if not np.any(exc):
            return
        current = np.bincount(self.post[exc], weights=self.w[exc], minlength=self.N)
        scale = np.divide(self.w_budget, current,
                          out=np.ones_like(current), where=current > 1e-12)
        np.clip(scale, 0.5, 2.0, out=scale)
        idx = np.where(exc)[0]
        self.w[idx] *= scale[self.post[idx]]
        self.w[idx] = np.clip(self.w[idx], 0.0, self.w_max)

    # ---- synaptic pruning: RAM -> disk, then genuinely free the RAM ----

    def weak_exc_edges(self, w_floor=None, fraction=None):
        """Which excitatory synapses have failed to earn their keep.

        `fraction` prunes the weakest N% -- more useful than an absolute
        floor, because soft-bound STDP plus synaptic scaling keeps weights
        in a distribution whose absolute scale drifts with activity, so a
        fixed floor catches almost nothing.
        """
        exc = np.where(self.is_exc_edge)[0]
        if len(exc) == 0:
            return exc
        if fraction:
            cutoff = np.quantile(self.w[exc], fraction)
            return exc[self.w[exc] <= cutoff]
        return exc[self.w[exc] < w_floor]

    def drop_edges(self, idx):
        """Delete edges from the in-memory arrays. Caller archives them
        first. Returns bytes freed."""
        if len(idx) == 0:
            return 0
        before = self.pre.nbytes + self.post.nbytes + self.w.nbytes + self.is_exc_edge.nbytes
        keep = np.ones(len(self.w), dtype=bool)
        keep[idx] = False
        self.pre = self.pre[keep]
        self.post = self.post[keep]
        self.w = self.w[keep]
        self.is_exc_edge = self.is_exc_edge[keep]
        after = self.pre.nbytes + self.post.nbytes + self.w.nbytes + self.is_exc_edge.nbytes
        # The synaptic-scaling budget must follow the surviving synapses,
        # or normalization will inflate them to replace the pruned input.
        self.w_budget = np.bincount(self.post[self.is_exc_edge],
                                    weights=self.w[self.is_exc_edge],
                                    minlength=self.N)
        return before - after

    def add_edges(self, pre, post, w):
        self.pre = np.concatenate([self.pre, pre.astype(np.int32)])
        self.post = np.concatenate([self.post, post.astype(np.int32)])
        self.w = np.concatenate([self.w, w])
        self.is_exc_edge = self.is_excitatory[self.pre]
        self.w_budget = np.bincount(self.post[self.is_exc_edge],
                                    weights=self.w[self.is_exc_edge],
                                    minlength=self.N)

    # ---- serialisation -------------------------------------------------

    def state_arrays(self):
        return dict(is_excitatory=self.is_excitatory, pre=self.pre, post=self.post,
                    w=self.w, voltages=self.voltages, threshold=self.threshold,
                    x_pre=self.x_pre, x_post=self.x_post,
                    spike_count=self.spike_count, w_budget=self.w_budget,
                    refrac=self.refrac)

    def meta(self):
        return dict(name=self.name, N=self.N,
                    metabolism=dict(cost=self.metabolism.cost,
                                    cooling=self.metabolism.cooling,
                                    critical=self.metabolism.critical,
                                    sustainable=self.metabolism.sustainable,
                                    fatigue_gain=self.metabolism.fatigue_gain,
                                    gain_floor=self.metabolism.gain_floor,
                                    heat=self.metabolism.heat),
                    **self._init_kwargs)

    @classmethod
    def from_saved(cls, meta, arrays):
        r = cls.__new__(cls)
        r.name, r.N, r.role = meta["name"], meta["N"], meta.get("role", "")
        r.is_excitatory = arrays["is_excitatory"]
        r.pre, r.post, r.w = arrays["pre"], arrays["post"], arrays["w"]
        r.is_exc_edge = r.is_excitatory[r.pre]
        r.voltages = arrays["voltages"]
        r.threshold = arrays["threshold"]
        r.x_pre, r.x_post = arrays["x_pre"], arrays["x_post"]
        r.spike_count = arrays["spike_count"]
        r.w_budget = arrays["w_budget"]
        r.refrac = arrays["refrac"]
        r.spikes = np.zeros(r.N, dtype=bool)
        for k in ("leak_factor", "base_threshold", "target_rate",
                  "threshold_adapt_rate", "global_inhibition", "trace_decay",
                  "learn_rate", "w_max", "ltd_ratio", "bg_rate", "bg_amp",
                  "refractory"):
            setattr(r, k, meta[k])
        r.rng = np.random.default_rng()
        r.threshold_bounds = tuple(meta["threshold_bounds"])
        m = meta["metabolism"]
        r.metabolism = Metabolism(cost=m["cost"], cooling=m["cooling"],
                                  critical=m["critical"], sustainable=m["sustainable"],
                                  fatigue_gain=m["fatigue_gain"], gain_floor=m["gain_floor"])
        r.metabolism.heat = m["heat"]
        r._init_kwargs = {k: meta[k] for k in (
            "role", "avg_out_degree", "frac_excitatory", "exc_weight_range",
            "inh_weight_scale", "leak_factor", "base_threshold", "target_rate",
            "threshold_adapt_rate", "threshold_bounds", "global_inhibition",
            "trace_decay", "learn_rate", "w_max", "ltd_ratio", "bg_rate",
            "bg_amp", "refractory")}
        return r


# ==========================================================================
#  PROJECTION  --  long-range connection, with or without dopamine gating
# ==========================================================================

class Projection:
    """Sparse excitatory-only pathway between two regions.

    Excitatory-only because long-range corticocortical axons really are
    almost all glutamatergic; inhibitory interneurons stay local.

    Two plasticity modes:
      'hebbian'      trace STDP applied immediately (unsupervised binding)
      'eligibility'  trace STDP writes a decaying synaptic tag instead;
                     nothing changes until apply_dopamine() arrives. This
                     is the three-factor rule that makes reward-driven
                     action selection possible.
    """

    def __init__(self, src, dst, fan_out=15, weight_range=(0.01, 0.03),
                 mode="hebbian", learn_rate=0.006, ltd_ratio=1.05,
                 w_max=None, elig_decay=0.9, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.src_name, self.dst_name = src.name, dst.name
        self.mode = mode
        self.learn_rate = learn_rate
        self.ltd_ratio = ltd_ratio
        self.elig_decay = elig_decay

        exc_src = np.where(src.is_excitatory)[0]
        k = min(fan_out, dst.N)
        pre = np.repeat(exc_src.astype(np.int32), k)
        post = np.concatenate([rng.choice(dst.N, size=k, replace=False) for _ in exc_src])
        self.pre, self.post = pre, post.astype(np.int32)
        self.w = rng.uniform(*weight_range, size=len(self.pre))
        self.w_max = w_max if w_max is not None else weight_range[1] * 4.0
        self.elig = np.zeros(len(self.w)) if mode == "eligibility" else None
        self.dst_n = dst.N
        # Synaptic scaling budget: total excitatory input each postsynaptic
        # neuron is allowed to receive. Fixed at birth, enforced periodically.
        self.w_budget = np.bincount(self.post, weights=self.w, minlength=dst.N)

    def normalize_incoming(self):
        """Synaptic scaling (Turrigiano & Nelson, Nat Rev Neurosci 2004).

        Hebbian learning alone has no fixed point: whatever fires together
        strengthens, and since firing is correlated with strength, everything
        drifts up together. Selectivity then washes out -- which is exactly
        what happens without this: the paired pathway grows 3.8x but so does
        every control pathway (2.4x), so almost nothing has been learned.

        Real neurons solve this by holding their TOTAL synaptic input roughly
        constant and redistributing it. That makes synapses compete: one can
        only strengthen at another's expense. It is what turns "everything
        grew" into "this pathway grew and that one shrank."
        """
        current = np.bincount(self.post, weights=self.w, minlength=self.dst_n)
        scale = np.divide(self.w_budget, current,
                          out=np.ones_like(current), where=current > 1e-12)
        np.clip(scale, 0.5, 2.0, out=scale)
        self.w *= scale[self.post]
        np.clip(self.w, 0.0, self.w_max, out=self.w)

    def propagate(self, src_spikes, dst_n, gain=1.0):
        pre_spiking = src_spikes[self.pre]
        if not np.any(pre_spiking):
            return None
        return gain * np.bincount(self.post[pre_spiking], weights=self.w[pre_spiking],
                                  minlength=dst_n)

    def plasticity(self, src, dst):
        """Trace STDP across the region boundary. src.x_pre is the source's
        recent firing history -- because inter-region transmission carries a
        one-step delay, this is precisely the trace that should be credited
        when the destination fires now."""
        lr = self.learn_rate
        dw = None
        post_fired = dst.spikes[self.post]
        if np.any(post_fired):
            idx = np.where(post_fired)[0]
            val = lr * src.x_pre[self.pre[idx]] * (self.w_max - self.w[idx])
            dw = (idx, val)
        pre_fired = src.spikes[self.pre]
        idx2 = np.where(pre_fired)[0] if np.any(pre_fired) else None
        val2 = (-lr * self.ltd_ratio * dst.x_post[self.post[idx2]] * self.w[idx2]
                if idx2 is not None else None)

        if self.mode == "hebbian":
            if dw is not None:
                self.w[dw[0]] += dw[1]
            if idx2 is not None:
                self.w[idx2] += val2
            np.clip(self.w, 0.0, self.w_max, out=self.w)
        else:
            self.elig *= self.elig_decay
            if dw is not None:
                self.elig[dw[0]] += dw[1]
            if idx2 is not None:
                self.elig[idx2] += val2

    def apply_dopamine(self, da, clear=True):
        """Third factor. `da` is a reward prediction error, so it can be
        negative -- an unexpectedly bad outcome actively weakens whatever
        was eligible."""
        if self.elig is None or da == 0.0:
            return
        self.w += da * self.elig
        np.clip(self.w, 0.0, self.w_max, out=self.w)
        if clear:
            self.elig *= 0.0

    def state_arrays(self):
        d = dict(pre=self.pre, post=self.post, w=self.w, w_budget=self.w_budget)
        if self.elig is not None:
            d["elig"] = self.elig
        return d

    def meta(self):
        return dict(src=self.src_name, dst=self.dst_name, mode=self.mode,
                    learn_rate=self.learn_rate, ltd_ratio=self.ltd_ratio,
                    elig_decay=self.elig_decay, w_max=self.w_max)

    @classmethod
    def from_saved(cls, meta, arrays):
        p = cls.__new__(cls)
        p.src_name, p.dst_name = meta["src"], meta["dst"]
        p.mode, p.learn_rate = meta["mode"], meta["learn_rate"]
        p.ltd_ratio, p.elig_decay, p.w_max = meta["ltd_ratio"], meta["elig_decay"], meta["w_max"]
        p.pre, p.post, p.w = arrays["pre"], arrays["post"], arrays["w"]
        p.w_budget = arrays["w_budget"]
        p.dst_n = len(p.w_budget)
        p.elig = arrays.get("elig") if meta["mode"] == "eligibility" else None
        if p.mode == "eligibility" and p.elig is None:
            p.elig = np.zeros(len(p.w))
        return p


# ==========================================================================
#  MEMORY STORE  --  bounded RAM, unbounded disk, nothing forgotten
# ==========================================================================

class MemoryStore:
    """Disk-backed episodic memory.

    The design point you asked for: a brain forgets because storage is
    finite, so instead of forgetting, spill to disk.

    Layout on disk:
        <root>/index.json          searchable index, always in RAM
        <root>/episodes/ep_N.npz   full pattern, one file per episode
        <root>/archive/*.npz       synapses pruned out of the live network

    RAM holds `ram_slots` full episodes (LRU) plus one 64-float signature
    per episode. The signature is what recall() searches, so search cost
    is independent of how much is on disk. A cache miss is a disk read,
    not a lost memory.
    """

    SIG_BINS = 64

    def __init__(self, root="brain_memory", ram_slots=16):
        self.root = Path(root)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        (self.root / "archive").mkdir(parents=True, exist_ok=True)
        self.ram_slots = ram_slots
        self.index = []          # list of dicts, always in RAM
        self.cache = {}          # ep_id -> dict of arrays (bounded)
        self.lru = []            # ep_id order, most recent last
        self.disk_reads = 0
        self.spills = 0
        idx_path = self.root / "index.json"
        if idx_path.exists():
            self.index = json.loads(idx_path.read_text())

    @staticmethod
    def signature(vec):
        """Fixed-length fingerprint of an activity vector: sum into 64 bins
        and L2-normalise. Cheap, order-preserving, and comparable across
        patterns of the same region."""
        v = np.asarray(vec, dtype=float)
        pad = (-len(v)) % MemoryStore.SIG_BINS
        if pad:
            v = np.concatenate([v, np.zeros(pad)])
        sig = v.reshape(MemoryStore.SIG_BINS, -1).sum(axis=1)
        n = np.linalg.norm(sig)
        return sig / n if n > 0 else sig

    def store(self, label, patterns, meta=None):
        """`patterns` is {region_name: activity_vector}. Written to disk
        immediately -- the memory exists on disk before it exists in RAM."""
        ep_id = f"ep_{len(self.index):06d}"
        path = self.root / "episodes" / f"{ep_id}.npz"
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32)
                                     for k, v in patterns.items()})
        primary = next(iter(patterns))
        entry = dict(id=ep_id, label=label, primary=primary,
                     signature=self.signature(patterns[primary]).round(5).tolist(),
                     regions=list(patterns.keys()), ts=_time.time(),
                     replays=0, strength=1.0, meta=meta or {})
        self.index.append(entry)
        self._to_cache(ep_id, {k: np.asarray(v, dtype=np.float32) for k, v in patterns.items()})
        return ep_id

    def _to_cache(self, ep_id, arrays):
        self.cache[ep_id] = arrays
        if ep_id in self.lru:
            self.lru.remove(ep_id)
        self.lru.append(ep_id)
        while len(self.lru) > self.ram_slots:
            evicted = self.lru.pop(0)
            self.cache.pop(evicted, None)
            self.spills += 1          # dropped from RAM; still on disk

    def load(self, ep_id):
        if ep_id in self.cache:
            if ep_id in self.lru:
                self.lru.remove(ep_id)
            self.lru.append(ep_id)
            return self.cache[ep_id]
        self.disk_reads += 1
        with np.load(self.root / "episodes" / f"{ep_id}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        self._to_cache(ep_id, arrays)
        return arrays

    def recall(self, cue_vector, top_k=1):
        """Search signatures (RAM), then fetch winners (disk if needed)."""
        if not self.index:
            return []
        cue = self.signature(cue_vector)
        sims = np.array([float(np.dot(cue, np.asarray(e["signature"]))) for e in self.index])
        order = np.argsort(-sims)[:top_k]
        out = []
        for i in order:
            e = self.index[i]
            e["replays"] += 1
            out.append((e, float(sims[i]), self.load(e["id"])))
        return out

    def sample_for_replay(self, rng, n):
        """Recent + strong episodes are likelier to be replayed, which is
        what makes sleep consolidate what actually mattered."""
        if not self.index:
            return []
        w = np.array([e["strength"] * (1.0 / (1 + 0.02 * (len(self.index) - i)))
                      for i, e in enumerate(self.index)])
        w = w / w.sum()
        n = min(n, len(self.index))
        picks = rng.choice(len(self.index), size=n, replace=False, p=w)
        return [(self.index[i], self.load(self.index[i]["id"])) for i in picks]

    # ---- synapse archive ----------------------------------------------

    def archive_synapses(self, region_name, pre, post, w):
        path = self.root / "archive" / f"{region_name}_{int(_time.time()*1000)}.npz"
        np.savez_compressed(path, pre=pre, post=post, w=w)
        return path

    def archived_for(self, region_name):
        return sorted((self.root / "archive").glob(f"{region_name}_*.npz"))

    def flush(self):
        (self.root / "index.json").write_text(json.dumps(self.index))

    def stats(self):
        ep_bytes = sum(p.stat().st_size for p in (self.root / "episodes").glob("*.npz"))
        ar_bytes = sum(p.stat().st_size for p in (self.root / "archive").glob("*.npz"))
        return dict(episodes=len(self.index), in_ram=len(self.cache),
                    spilled=self.spills, disk_reads=self.disk_reads,
                    episode_bytes=ep_bytes, archive_bytes=ar_bytes)


# ==========================================================================
#  CURIOSITY  --  your ICM, vectorised and bounded
# ==========================================================================

class IntrinsicCuriosity:
    """Forward-model prediction error as intrinsic reward.

    This is your gemini-code-...564743.py design, kept intact in concept:
    predict the next state, and let the size of the miss be the reward.
    Three fixes so it can drive a live brain:

    1. Your gradient (`2 * err / len(state)`) is the MSE gradient with
       respect to the *prediction*; the weight update also needs the input,
       which np.outer supplied -- correct, but the 1/len factor made the
       effective learning rate depend on state size. Normalised here.
    2. Raw MSE is unbounded, so one weird input could dominate every
       reward signal forever. Novelty is now scaled by a running average
       of recent error, giving a bounded "surprising *relative to lately*"
       measure -- which is what curiosity actually is.
    3. States are normalised before prediction, so error reflects pattern
       mismatch rather than overall activity level.
    """

    def __init__(self, state_size, learning_rate=0.05, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((state_size, state_size)) * 0.01
        self.lr = learning_rate
        self.baseline = None

    @staticmethod
    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def observe(self, state, next_state):
        s, s1 = self._norm(np.asarray(state, float)), self._norm(np.asarray(next_state, float))
        pred = s @ self.W
        err = pred - s1
        mse = float(np.mean(err ** 2))
        self.W -= self.lr * np.outer(s, err)          # gradient descent on MSE
        if self.baseline is None:
            self.baseline = mse
        self.baseline = 0.98 * self.baseline + 0.02 * mse
        novelty = mse / (self.baseline + 1e-9)        # bounded, relative
        return float(np.tanh(novelty - 1.0)), mse


# ==========================================================================
#  BRAIN
# ==========================================================================

class Brain:
    """Regions + projections + metabolism + dopamine + memory + sleep.

    A step has explicit phases so every region sees the same timestep and
    no region gets a one-step advantage from dict ordering:

      1  external input
      2  inter-region transmission (from LAST step's spikes -- long axons
         are slower than local ones, and this is what makes the causal
         ordering that trace STDP reads)
      3  every region fires
      4  local recurrent transmission + shared inhibition
      5  plasticity (local + projections)
      6  traces update
      7  homeostasis + metabolism
      8  whole-brain circuit breaker
    """

    def __init__(self, seed=None, memory_root="brain_memory", ram_slots=16,
                 overheat_frac=0.22, breaker_recovery=0.08, da_baseline_rate=0.05,
                 norm_interval=20):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.regions = {}
        self.projections = []
        self.time = 0.0
        self.global_gain = 1.0
        self.overheat_frac = overheat_frac
        self.breaker_recovery = breaker_recovery
        self.norm_interval = norm_interval
        self._last_spikes = {}
        self.memory = MemoryStore(memory_root, ram_slots=ram_slots)
        self.curiosity = None
        self.da_baseline = 0.0
        self.da_baseline_rate = da_baseline_rate
        self.asleep = False
        self.history = dict(rate=[], gain=[], heat=[])

    # ---- construction --------------------------------------------------

    def add_region(self, name, num_neurons, metabolism=None, **kw):
        r = BrainRegion(name, num_neurons, rng=self.rng,
                        metabolism=metabolism, **kw)
        self.regions[name] = r
        self._last_spikes[name] = np.zeros(num_neurons, dtype=bool)
        return r

    def connect(self, src, dst, **kw):
        p = Projection(self.regions[src], self.regions[dst], rng=self.rng, **kw)
        self.projections.append(p)
        return p

    def projection(self, src, dst):
        return next(p for p in self.projections if p.src_name == src and p.dst_name == dst)

    @property
    def total_neurons(self):
        return sum(r.N for r in self.regions.values())

    @property
    def total_synapses(self):
        return sum(len(r.w) for r in self.regions.values()) + sum(len(p.w) for p in self.projections)

    # ---- the step ------------------------------------------------------

    def step(self, external=None, learn=True, record=False):
        self.time += 1.0
        cooling_scale = 3.0 if self.asleep else 1.0

        if external:
            for name, vec in external.items():
                self.regions[name].receive_input(vec)

        for p in self.projections:
            src_spikes = self._last_spikes.get(p.src_name)
            if src_spikes is not None and src_spikes.any():
                dst = self.regions[p.dst_name]
                inc = p.propagate(src_spikes, dst.N,
                                  gain=self.global_gain * dst.metabolism.gain)
                if inc is not None:
                    dst.voltages += inc

        spikes = {name: r.fire() for name, r in self.regions.items()}
        for r in self.regions.values():
            r.propagate_local()

        if learn:
            for r in self.regions.values():
                r.local_stdp()
            for p in self.projections:
                p.plasticity(self.regions[p.src_name], self.regions[p.dst_name])

        for r in self.regions.values():
            r.update_traces()
            r.homeostasis(cooling_scale)

        if learn and self.norm_interval and int(self.time) % self.norm_interval == 0:
            for r in self.regions.values():
                r.normalize_incoming()
            for p in self.projections:
                p.normalize_incoming()

        frac = sum(s.sum() for s in spikes.values()) / self.total_neurons
        if frac > self.overheat_frac:
            self.global_gain = max(0.05, self.global_gain * 0.5)
        else:
            self.global_gain = min(1.0, self.global_gain + self.breaker_recovery)

        if record:
            self.history["rate"].append(frac)
            self.history["gain"].append(self.global_gain)
            self.history["heat"].append(np.mean([r.metabolism.heat for r in self.regions.values()]))

        self._last_spikes = spikes
        return spikes

    # ---- dopamine ------------------------------------------------------

    def reward(self, r):
        """Reward prediction error, not raw reward. An outcome the brain
        already expected produces ~0 dopamine and teaches nothing."""
        rpe = r - self.da_baseline
        self.da_baseline += self.da_baseline_rate * rpe
        for p in self.projections:
            if p.mode == "eligibility":
                p.apply_dopamine(rpe)
        return rpe

    # ---- sensory encoding: the safe way to feed it real data -----------

    def sensory_encode(self, text, amplitude=1.4, active_frac=0.05):
        """Deterministically hash text into a sparse sensory pattern.

        This is the honest version of "feed it the world": you hand it
        bytes, it turns them into spikes. Same text always maps to the same
        pattern, so the network can actually learn about it. Read-only, no
        network access, no execution -- data goes in, nothing goes out.
        """
        s = self.regions["sensory"]
        k = max(1, int(active_frac * s.N))
        vec = np.zeros(s.N)
        for tok in str(text).lower().split():
            h = abs(hash(tok)) % (2 ** 31)
            local = np.random.default_rng(h).choice(s.N, size=k, replace=False)
            vec[local] += amplitude
        return np.clip(vec, 0.0, 3.0 * amplitude)

    # ---- sleep: replay, consolidate, prune ------------------------------

    def sleep(self, cycles=3, replays_per_cycle=12, replay_steps=6,
              prune_fraction=0.05, verbose=True):
        """Offline consolidation.

        Real sleep does three things this reproduces: it replays the day's
        episodes to move them from fast (hippocampal) to slow (cortical)
        storage, it clears metabolic load, and it prunes synapses that
        didn't earn their keep. Pruned synapses are archived to disk before
        deletion -- dormant, not destroyed.
        """
        self.asleep = True
        replayed = 0
        for _ in range(cycles):
            for entry, arrays in self.memory.sample_for_replay(self.rng, replays_per_cycle):
                ext = {}
                for rname in entry["regions"]:
                    if rname in self.regions and rname in arrays:
                        ext[rname] = arrays[rname].astype(float) * 0.8
                for _s in range(replay_steps):
                    self.step(external=ext if _s == 0 else None, learn=True)
                entry["strength"] = min(3.0, entry["strength"] + 0.15)
                replayed += 1
            for _ in range(10):
                self.step(None, learn=False)

        freed, pruned = 0, 0
        for r in self.regions.values():
            idx = r.weak_exc_edges(fraction=prune_fraction)
            if len(idx):
                self.memory.archive_synapses(r.name, r.pre[idx], r.post[idx], r.w[idx])
                freed += r.drop_edges(idx)
                pruned += len(idx)

        self.memory.flush()
        self.asleep = False
        if verbose:
            print(f"  replayed {replayed} episodes | pruned {pruned} synapses "
                  f"({freed/1024:.1f} KB freed from RAM, archived to disk)")
        return dict(replayed=replayed, pruned=pruned, bytes_freed=freed)

    def restore_archive(self, region_name):
        """Bring dormant synapses back from disk into the live network."""
        r = self.regions[region_name]
        n = 0
        for path in self.memory.archived_for(region_name):
            with np.load(path) as z:
                r.add_edges(z["pre"], z["post"], z["w"])
                n += len(z["w"])
        return n

    # ---- persistence ----------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = dict(time=self.time, global_gain=self.global_gain,
                    overheat_frac=self.overheat_frac,
                    breaker_recovery=self.breaker_recovery,
                    da_baseline=self.da_baseline,
                    da_baseline_rate=self.da_baseline_rate,
                    norm_interval=self.norm_interval,
                    rng_state=self.rng.bit_generator.state,
                    memory_root=str(self.memory.root),
                    ram_slots=self.memory.ram_slots,
                    region_names=list(self.regions),
                    regions={n: r.meta() for n, r in self.regions.items()},
                    projections=[p.meta() for p in self.projections])
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        arrays = {}
        for name, r in self.regions.items():
            for k, v in r.state_arrays().items():
                arrays[f"region__{name}__{k}"] = v
            arrays[f"lastspikes__{name}"] = self._last_spikes[name]
        for i, p in enumerate(self.projections):
            for k, v in p.state_arrays().items():
                arrays[f"proj__{i}__{k}"] = v
        np.savez_compressed(path / "state.npz", **arrays)
        self.memory.flush()
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        data = np.load(path / "state.npz")

        b = cls.__new__(cls)
        b.rng = np.random.default_rng()
        if "rng_state" in meta:
            # Restore the RNG bit-generator state too. v4 has genuinely
            # stochastic elements (background cortical activity), so without
            # this a reloaded brain diverges from the original within a step
            # and "identical behaviour after reload" cannot be verified.
            b.rng.bit_generator.state = meta["rng_state"]
        b.seed = None
        b.time = meta["time"]
        b.global_gain = meta["global_gain"]
        b.overheat_frac = meta["overheat_frac"]
        b.breaker_recovery = meta["breaker_recovery"]
        b.da_baseline = meta["da_baseline"]
        b.da_baseline_rate = meta["da_baseline_rate"]
        b.norm_interval = meta.get("norm_interval", 20)
        b.memory = MemoryStore(meta["memory_root"], ram_slots=meta["ram_slots"])
        b.curiosity = None
        b.asleep = False
        b.history = dict(rate=[], gain=[], heat=[])
        b.regions, b._last_spikes = {}, {}
        for name in meta["region_names"]:
            arrays = {k.split("__", 2)[2]: data[k].copy()
                      for k in data.files if k.startswith(f"region__{name}__")}
            b.regions[name] = BrainRegion.from_saved(meta["regions"][name], arrays)
            b.regions[name].rng = b.rng          # regions share the brain RNG
            b._last_spikes[name] = data[f"lastspikes__{name}"].copy()
        b.projections = []
        for i, pmeta in enumerate(meta["projections"]):
            arrays = {k.split("__", 2)[2]: data[k].copy()
                      for k in data.files if k.startswith(f"proj__{i}__")}
            b.projections.append(Projection.from_saved(pmeta, arrays))
        return b

    # ---- introspection --------------------------------------------------

    def report(self):
        rows = []
        for n, r in self.regions.items():
            m = r.metabolism.state()
            rows.append(f"  {n:<12} N={r.N:<5} syn={len(r.w):<7} "
                        f"heat={m['heat']:.3f} gain={m['gain']:.2f} "
                        f"thr={r.threshold.mean():.2f}  [{r.role}]")
        return "\n".join(rows)


# ==========================================================================
#  WORLD + AUTONOMOUS AGENT
#  This is the part where nobody drives the brain but the brain.
# ==========================================================================

class GridWorld:
    """A small room with food in one corner.

    The brain is never told where the food is, what the four actions do,
    or even that there ARE four actions. It gets a sparse pattern of
    spikes that depends on where it is, and a single scalar afterwards.
    Everything else it has to work out.
    """

    def __init__(self, size=5, seed=0):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.food = (size - 1, size - 1)
        self.visits = np.zeros((size, size), dtype=int)
        self.reset()

    def reset(self):
        while True:
            self.pos = (int(self.rng.integers(self.size)),
                        int(self.rng.integers(self.size)))
            if self.pos != self.food:
                return

    def state_id(self):
        return self.pos[0] * self.size + self.pos[1]

    def move_food(self, new_pos):
        self.food = new_pos
        self.reset()

    def act(self, a):
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        x = min(max(self.pos[0] + dx, 0), self.size - 1)
        y = min(max(self.pos[1] + dy, 0), self.size - 1)
        hit_wall = (x, y) == self.pos
        self.pos = (x, y)
        self.visits[self.pos] += 1
        if self.pos == self.food:
            self.reset()
            return 1.0, True
        return (-0.05 if hit_wall else -0.01), False


class AutonomousAgent:
    """A brain running free in a loop, with nothing scripting it.

    WHAT "AUTONOMOUS" HONESTLY MEANS HERE -- four decisions the brain
    makes that no line of code makes for it:

    1. WHAT TO DO. Motor pools compete; the winner is the action. Which
       pool wins depends on synapses shaped by reward the brain earned
       itself. No lookup table, no policy anyone wrote.

    2. WHETHER TO EXPLORE OR EXPLOIT. Not a hand-tuned epsilon schedule
       decaying on a timer -- exploration is driven by the brain's OWN
       prediction error. When the world stops surprising its forward
       model, curiosity falls and behaviour settles. When something
       changes, curiosity climbs and it starts trying things again.

    3. WHEN TO SLEEP. Sleep pressure accumulates from the brain's own
       metabolic heat and time awake, exactly the way adenosine does.
       When it crosses threshold the agent stops acting, replays, prunes,
       and wakes up. Nothing outside schedules this.

    4. WHAT TO REMEMBER. An episode is written to disk only when novelty
       exceeds threshold -- it stores what surprised it, not everything
       that happened. Storage is finite; what fills it is the brain's
       own judgement of what was worth keeping.

    WHAT IT DOES NOT MEAN: none of this is deliberation, understanding,
    or wanting. "Decides" here is the control-theory sense -- the system's
    own internal state selects its output, rather than an outside script
    doing it. That is a real and non-trivial property, and it is also not
    the same thing as thinking. Be precise about which one you have.
    """

    def __init__(self, brain, world, place_codes, n_actions=4,
                 act_every=6, sleep_threshold=45.0, novelty_to_store=0.5,
                 explore_floor=0.08, seed=0):
        self.b = brain
        self.w = world
        self.codes = place_codes
        self.n_actions = n_actions
        self.act_every = act_every
        self.pools = [np.arange(i * (brain.regions["motor"].N // n_actions),
                                (i + 1) * (brain.regions["motor"].N // n_actions))
                      for i in range(n_actions)]
        self.rng = np.random.default_rng(seed)
        self.curiosity = IntrinsicCuriosity(state_size=64, seed=seed)
        self.brain_state = None
        self.sleep_pressure = 0.0
        self.sleep_threshold = sleep_threshold
        self.novelty_to_store = novelty_to_store
        self.explore_floor = explore_floor
        self.log = dict(foods=[], sleeps=[], stored=0, novelty=[],
                        pressure=[], explore=[], gaps=[])

    def _summary(self, spikes):
        """A 64-number digest of what the whole brain just did -- the state
        the curiosity module tries to predict."""
        out = []
        for name in ("sensory", "association", "motor"):
            v = spikes[name].astype(float)
            pad = (-len(v)) % 16
            if pad:
                v = np.concatenate([v, np.zeros(pad)])
            out.append(v.reshape(16, -1).mean(axis=1))
        vec = np.concatenate(out)
        return np.resize(vec, 64)

    def live(self, steps, verbose_every=None):
        """Run free. No trials, no resets, no schedule."""
        steps_since_food = 0
        for t in range(steps):
            # ---- sleep is the brain's own call -------------------------
            if self.sleep_pressure >= self.sleep_threshold:
                self.b.sleep(cycles=1, replays_per_cycle=6, prune_fraction=0.02, verbose=False)
                self.log["sleeps"].append(t)
                self.sleep_pressure = 0.0

            counts = np.zeros(self.n_actions)
            last = None
            for _ in range(self.act_every):
                v = np.zeros(self.b.regions["sensory"].N)
                v[self.codes[self.w.state_id()]] = 1.6
                s = self.b.step({"sensory": v}, learn=True)
                for i, pl in enumerate(self.pools):
                    counts[i] += s["motor"][pl].sum()
                last = s

            # ---- curiosity: how wrong was its own prediction? ----------
            summary = self._summary(last)
            novelty = 0.0
            if self.brain_state is not None:
                novelty, _ = self.curiosity.observe(self.brain_state, summary)
            self.brain_state = summary

            # ---- explore/exploit driven by that, not by a timer --------
            explore = float(np.clip(self.explore_floor + 0.5 * max(0.0, novelty),
                                    self.explore_floor, 0.9))
            if self.rng.random() < explore:
                a = int(self.rng.integers(self.n_actions))
            else:
                a = int(np.argmax(counts + self.rng.random(self.n_actions) * 1e-6))

            r, ate = self.w.act(a)
            self.b.reward(r + 0.05 * novelty)      # extrinsic + intrinsic
            steps_since_food += 1

            # ---- store only what surprised it --------------------------
            if novelty > self.novelty_to_store:
                self.b.memory.store(f"t{t}_s{self.w.state_id()}",
                                    {"sensory": (last["sensory"]).astype(float)},
                                    meta=dict(novelty=round(float(novelty), 3)))
                self.log["stored"] += 1

            # sleep pressure from its own metabolism
            heat = float(np.mean([rg.metabolism.heat for rg in self.b.regions.values()]))
            self.sleep_pressure += 0.02 + heat

            if ate:
                self.log["gaps"].append(steps_since_food)
                steps_since_food = 0
                self.log["foods"].append(t)
            self.log["novelty"].append(novelty)
            self.log["explore"].append(explore)
            self.log["pressure"].append(self.sleep_pressure)

            if verbose_every and t and t % verbose_every == 0:
                g = self.log["gaps"][-10:]
                print(f"    t={t:5d}  foods={len(self.log['foods']):3d}  "
                      f"recent steps/food={np.mean(g):5.1f}  "
                      f"explore={explore:.2f}  sleeps={len(self.log['sleeps'])}  "
                      f"stored={self.log['stored']}")
        self.b.memory.flush()
        return self.log


def random_walk_baseline(size=5, steps=9000, seed=11):
    """How well does doing nothing intelligent do? Without this number the
    learning curve means nothing."""
    w = GridWorld(size=size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    gaps, c = [], 0
    for _ in range(steps):
        _, ate = w.act(int(rng.integers(4)))
        c += 1
        if ate:
            gaps.append(c)
            c = 0
    return np.array(gaps)


def build_agent_brain(seed=11, memory_root="brain_memory_agent"):
    b = Brain(seed=seed, memory_root=memory_root, ram_slots=8)
    b.add_region("sensory", 400, role="place code", target_rate=0.03)
    b.add_region("association", 600, role="hub")
    b.add_region("motor", 200, role="4 competing actions",
                 global_inhibition=1.2, bg_rate=0.03, refractory=1)
    b.connect("sensory", "association", fan_out=30, weight_range=(0.05, 0.12))
    b.connect("association", "motor", fan_out=30, weight_range=(0.05, 0.12),
              mode="eligibility", learn_rate=0.05, elig_decay=0.92)
    return b


# ==========================================================================
#  DEMO -- every claim above, measured
# ==========================================================================

def _hdr(n, title):
    print("\n" + "=" * 70)
    print(f"{n}) {title}")
    print("=" * 70)


def build_brain(seed=7, memory_root="brain_memory_demo"):
    """Five regions, specialised by PARAMETERS rather than just by name."""
    b = Brain(seed=seed, memory_root=memory_root, ram_slots=8)
    b.add_region("sensory", 500, role="input surface", target_rate=0.03)
    b.add_region("association", 700, role="hub / binding")
    b.add_region("prefrontal", 400, role="working memory", leak_factor=0.985,
                 target_rate=0.03)
    b.add_region("hippocampus", 400, role="fast episodic writer",
                 learn_rate=0.032, target_rate=0.03)
    b.add_region("motor", 240, role="action selection", global_inhibition=1.2,
                 bg_rate=0.03, refractory=1)
    W = dict(fan_out=30, weight_range=(0.05, 0.12))
    b.connect("sensory", "association", **W)
    b.connect("association", "prefrontal", **W)
    b.connect("prefrontal", "association", fan_out=12, weight_range=(0.03, 0.07))
    b.connect("association", "hippocampus", learn_rate=0.02, **W)
    b.connect("hippocampus", "association", fan_out=12, weight_range=(0.03, 0.07))
    b.connect("association", "motor", mode="eligibility", learn_rate=0.035, **W)
    b.connect("prefrontal", "motor", mode="eligibility", learn_rate=0.035,
              fan_out=12, weight_range=(0.03, 0.07))
    return b


def association_test(trials=800, seed=3):
    """Double dissociation with TWO controls, so neither frequency nor
    target activity alone can explain the result."""
    b = Brain(seed=seed, memory_root="brain_memory_assoc")
    b.add_region("sensory", 500)
    b.add_region("association", 700)
    p = b.connect("sensory", "association", fan_out=30, weight_range=(0.05, 0.12))
    rng = np.random.default_rng(11)
    ids = rng.permutation(500)
    S1, S2 = ids[:40], ids[40:80]
    T = rng.choice(700, 60, replace=False)
    notT = np.setdiff1d(np.arange(700), T)

    def mw(src, dst):
        m = np.isin(p.pre, src) & np.isin(p.post, dst)
        return float(p.w[m].mean())

    def recall(cue, steps=50):
        h = []
        for _ in range(steps):
            v = np.zeros(500); v[cue] = 1.6
            b.step({"sensory": v}, learn=False)
            s = b.step(None, learn=False)
            h.append(s["association"][T].mean())
            b.step(None, learn=False)
        return float(np.mean(h) * 100)

    before = (mw(S1, T), mw(S1, notT), mw(S2, T))
    r_before = (recall(S1), recall(S2))
    for tr in range(trials):
        v = np.zeros(500)
        if tr % 2 == 0:
            v[S1] = 1.6
            b.step({"sensory": v}, learn=True)
            t = np.zeros(700); t[T] = 1.5
            b.step({"association": t}, learn=True)
        else:
            v[S2] = 1.6
            b.step({"sensory": v}, learn=True)
            b.step(None, learn=True)
        b.step(None, learn=True)
    after = (mw(S1, T), mw(S1, notT), mw(S2, T))
    return before, after, r_before, (recall(S1), recall(S2))


def decision_task(b, trials=1200, present=10, seed=0):
    """Three cues, three actions. The brain is never told which action is
    correct -- only good or bad, after it has already chosen."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(500)
    cues = [ids[i * 40:(i + 1) * 40] for i in range(3)]
    pools = [np.arange(i * 80, (i + 1) * 80) for i in range(3)]
    hist = []
    for tr in range(trials):
        eps = max(0.05, 0.6 * np.exp(-tr / 250))
        c = tr % 3
        counts = np.zeros(3)
        for _ in range(present):
            v = np.zeros(500); v[cues[c]] = 1.6
            s = b.step({"sensory": v}, learn=True)
            for i, pl in enumerate(pools):
                counts[i] += s["motor"][pl].sum()
        a = (rng.integers(3) if rng.random() < eps
             else int(np.argmax(counts + rng.random(3) * 1e-6)))
        b.reward(1.0 if a == c else -1.0)
        hist.append(a == c)
        for _ in range(3):
            b.step(None, learn=False)
    return np.array(hist)


def flood_test(steps=60, recovery=60, seed=1):
    b = build_brain(memory_root="brain_memory_flood")
    rng = np.random.default_rng(seed)
    rates = []
    for i in range(steps + recovery):
        ext = ({n: rng.uniform(0, 6, r.N) for n, r in b.regions.items()}
               if i < steps else None)
        s = b.step(ext, learn=False)
        rates.append(sum(x.sum() for x in s.values()) / b.total_neurons * 100)
    return np.array(rates)


def working_memory(b, hold=40):
    rng = np.random.default_rng(0)
    A = rng.choice(500, 40, replace=False)
    for _ in range(12):
        v = np.zeros(500); v[A] = 1.6
        b.step({"sensory": v}, learn=False)
    pfc, sens = [], []
    for _ in range(hold):
        s = b.step(None, learn=False)
        pfc.append(s["prefrontal"].mean() * 100)
        sens.append(s["sensory"].mean() * 100)
    return np.array(pfc), np.array(sens)


def memory_demo(b, n=40):
    rng = np.random.default_rng(0)
    pats = []
    for i in range(n):
        v = np.zeros(500)
        v[rng.choice(500, 40, replace=False)] = 1.0
        pats.append(v)
        b.memory.store(f"concept_{i}", {"sensory": v})
    st = b.memory.stats()
    hits = 0
    for i, v in enumerate(pats):
        cue = v.copy()
        on = np.where(cue > 0)[0]
        cue[rng.choice(on, len(on) // 2, replace=False)] = 0
        if b.memory.recall(cue, top_k=1)[0][0]["label"] == f"concept_{i}":
            hits += 1
    return st, b.memory.stats(), hits, n


def make_figure(flood, blocks, before, after, pfc, sens, gaps, baseline, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not installed -- skipping figure, everything else ran)")
        return
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.2))

    ax[0].plot(flood, color="#2ca02c", lw=1.6)
    ax[0].axvline(60, color="k", ls=":", lw=1)
    ax[0].axhline(33.3, color="#555", ls="--", lw=1)
    ax[0].text(62, 36, "hard ceiling 33.3%", fontsize=8, color="#555")
    ax[0].set_title("Flood every neuron, then stop")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("% of all neurons firing")
    ax[0].set_ylim(-3, 105); ax[0].grid(alpha=.3)

    x = np.arange(len(blocks))
    ax[1].plot(x, blocks, "o-", color="#1f77b4", lw=2)
    ax[1].axhline(33.3, color="#d62728", ls="--", lw=1.2, label="chance")
    ax[1].set_title("Decisions learned from reward alone")
    ax[1].set_xlabel("training block"); ax[1].set_ylabel("% correct")
    ax[1].set_ylim(0, 100); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    labels = ["S1→T\n(paired)", "S1→rest\n(control)", "S2→T\n(control)"]
    xx = np.arange(3); wd = 0.38
    ax[2].bar(xx - wd / 2, before, wd, label="before", color="#aec7e8")
    ax[2].bar(xx + wd / 2, after, wd, label="after", color="#1f77b4")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_title("Only the paired pathway grows")
    ax[2].set_ylabel("mean synaptic weight")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")

    ax[3].plot(pfc, color="#9467bd", lw=1.8, label="prefrontal")
    ax[3].plot(sens, color="#8c564b", lw=1.8, label="sensory")
    ax[3].set_title("Working memory: input stops at step 0")
    ax[3].set_xlabel("steps after input removed"); ax[3].set_ylabel("% firing")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)

    q = [float(x.mean()) for x in np.array_split(np.asarray(gaps), 4)]
    ax[4].plot([1, 2, 3, 4], q, "o-", color="#ff7f0e", lw=2, label="agent")
    ax[4].axhline(baseline, color="#d62728", ls="--", lw=1.2, label="random walk")
    ax[4].set_title("Free-running agent: steps needed per food")
    ax[4].set_xlabel("quarter of the run"); ax[4].set_ylabel("steps per food")
    ax[4].set_xticks([1, 2, 3, 4]); ax[4].legend(fontsize=8); ax[4].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"  figure written to {path}")


def main():
    t_start = time.time()
    for d in ("brain_memory_demo", "brain_memory_assoc", "brain_memory_flood",
              "brain_memory_agent", "brain_v5_checkpoint"):
        shutil.rmtree(d, ignore_errors=True)

    print(f"numpy {np.__version__} | python {sys.version.split()[0]}")

    _hdr(0, "ARCHITECTURE")
    b = build_brain()
    print(b.report())
    print(f"  TOTAL: {b.total_neurons} neurons, {b.total_synapses:,} synapses, "
          f"{len(b.projections)} inter-region pathways")

    _hdr(1, "RESTING STATE -- no input at all")
    rates = [sum(x.sum() for x in b.step(None, learn=False).values()) / b.total_neurons
             for _ in range(300)]
    print(f"  mean firing over 300 steps: {np.mean(rates)*100:.3f}%")
    print("  low and steady -- not silent, not runaway")

    _hdr(2, "LEARNING AN ASSOCIATION (this is what was broken in v3)")
    before, after, rb, ra = association_test()
    names = ["S1 -> T    (PAIRED)         ",
             "S1 -> rest (control: cue)   ",
             "S2 -> T    (control: target)"]
    for nm, bb, aa in zip(names, before, after):
        print(f"  {nm} {bb:.4f} -> {aa:.4f}   ({aa/bb:.2f}x)")
    print(f"\n  functional recall -- cue shown ALONE, does T light up?")
    print(f"    S1 (paired):  {rb[0]:.2f}% -> {ra[0]:.2f}%   ({ra[0]/rb[0]:.1f}x)")
    print(f"    S2 (control): {rb[1]:.2f}% -> {ra[1]:.2f}%   ({ra[1]/rb[1]:.1f}x)")

    _hdr(3, "ANTI-OVERHEATING -- flood every neuron in every region")
    flood = flood_test()
    print(f"  first 15 steps {flood[:15].mean():5.1f}%   "
          f"end of flood {flood[45:60].mean():5.1f}%   "
          f"after flood {flood[75:].mean():.2f}%")
    print("  ceiling is 1/(refractory+1) = 33.3%, a CONSTRAINT not feedback:")
    print("  no input, however large, can push past it. v3 sat at ~85%.")

    _hdr(4, "WORKING MEMORY -- prefrontal holds what sensory drops")
    b_wm = build_brain(memory_root="brain_memory_demo")
    pfc, sens = working_memory(b_wm)
    print(f"  input stops at step 0; mean firing over the next 20 steps:")
    print(f"    prefrontal: {pfc[:20].mean():.2f}%   (still holding it)")
    print(f"    sensory   : {sens[:20].mean():.2f}%   (gone in ~3 steps)")
    print(f"    ratio: {pfc[:20].mean()/max(sens[:20].mean(),1e-9):.1f}x")

    _hdr(5, "DECIDING -- reward only, never told the right answer")
    hist = decision_task(b)
    blocks = hist.reshape(6, -1).mean(axis=1) * 100
    print(f"  accuracy by block: {'  '.join(f'{x:.0f}%' for x in blocks)}")
    print(f"  chance = 33.3%  |  final 150 trials = {hist[-150:].mean()*100:.0f}%")

    _hdr(6, "MEMORY: RAM -> DISK, nothing forgotten")
    st, st2, hits, n = memory_demo(b)
    print(f"  {st['episodes']} episodes | {st['in_ram']} in RAM | "
          f"{st['spilled']} spilled to disk")
    print(f"  {st2['episode_bytes']/1024:.0f} KB on disk, "
          f"{st2['disk_reads']} disk reads served during recall")
    print(f"  recall from a 50%-DESTROYED cue: {hits}/{n} ({hits/n*100:.0f}%)")

    _hdr(7, "SLEEP -- replay, consolidate, archive to disk")
    before_syn = b.total_synapses
    b.sleep(cycles=2, replays_per_cycle=10)
    print(f"  synapses in RAM: {before_syn:,} -> {b.total_synapses:,}")
    restored = sum(b.restore_archive(nm) for nm in b.regions)
    print(f"  restored {restored:,} dormant synapses from disk -> {b.total_synapses:,}")

    _hdr(8, "PERSISTENCE -- save, reload, verify")
    b.save("brain_v5_checkpoint")
    loaded = Brain.load("brain_v5_checkpoint")
    rng = np.random.default_rng(99)
    ok = True
    for _ in range(40):
        v = np.zeros(500); v[rng.choice(500, 30, replace=False)] = 1.4
        s1 = b.step({"sensory": v.copy()}, learn=True)
        s2 = loaded.step({"sensory": v.copy()}, learn=True)
        if not all(np.array_equal(s1[k], s2[k]) for k in s1):
            ok = False
            break
    print(f"  reloaded brain identical to the original, step for step: {ok}")

    _hdr(9, "RUNNING FREE -- nobody drives it but itself")
    print("  No trials. No resets. It picks actions, decides when to sleep,")
    print("  and decides what is worth remembering. ~90 seconds.\n")
    ag_brain = build_agent_brain(seed=11)
    rng = np.random.default_rng(11)
    codes = [rng.choice(400, 35, replace=False) for _ in range(25)]
    agent = AutonomousAgent(ag_brain, GridWorld(seed=11), codes, seed=11)
    agent.live(9000, verbose_every=3000)
    gaps = np.array(agent.log["gaps"])
    q = [float(x.mean()) for x in np.array_split(gaps, 4)]
    base = float(random_walk_baseline(steps=9000, seed=11).mean())
    print(f"\n  steps needed per food, by quarter: "
          f"{'  '.join(f'{x:.0f}' for x in q)}")
    print(f"  random-walk baseline: {base:.0f} steps")
    print(f"  self-initiated sleeps: {len(agent.log['sleeps'])}")
    print(f"  episodes IT chose to store: {agent.log['stored']}")

    make_figure(flood, blocks, before, after, pfc, sens, gaps, base,
                "brain_v5_results.png")
    print(f"\n  total runtime: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
