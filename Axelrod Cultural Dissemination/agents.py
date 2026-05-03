import numpy as np
from mesa import Agent
 
class CulturalAgent(Agent):
    """A site (fixed location) with a cultural feature vector.
 
    Attributes:
        culture: numpy int array of length F (model.num_features). Each entry
                 is a trait index in [0, q) where q = model.num_traits.
    """
 
    def __init__(self, model):
        """Initialize an agent with a randomly assigned culture.
 
        Each of the F features is independently drawn uniformly from
        {0, 1, ..., q-1}. cultures are "randomly assigned" , 
        use independent uniform sampling.
 
        Use model.random so the entire run is reproducible from the model seed.
        """
        super().__init__(model)

        self.culture = np.array(
            [model.random.randrange(model.num_traits)
             for _ in range(model.num_features)],
            dtype=np.int8,
        )
 
    def cultural_similarity(self, other):
        """Proportion of features on which self and other share a trait.
 
       "the percentage of their features that have the identical trait." 
       Returns a float in [0.0, 1.0].
        """
        matches = int(np.sum(self.culture == other.culture))
        return matches / self.model.num_features
 
    def interact(self):
        """One interaction event for this agent (the active site).
 
        Axelrod's two-step rule :
            Step 1 (model-level): pick an active site at random, pick one
                of its neighbors at random. The model has already chosen
                `self` as the active site before calling this method;
                we pick the neighbor here.
            Step 2: with probability equal to cultural similarity, the
                two sites interact. Interaction = pick a random feature
                on which they differ and copy the neighbor's trait into
                the active site.
 
        It is the active site (self) whose culture changes, 
        not the neighbor's. Axelrod does this so edge sites have
        the same probability of being influenced as interior sites.
        """
        # Step 1 (neighbor selection)
        neighbors = self.model.get_neighbors(self.pos)
        if not neighbors:
            return
        neighbor = self.random.choice(neighbors)
 
        # Step 2: probabilistic interaction
        similarity = self.cultural_similarity(neighbor)
 
        # Two early-exit cases. Axelrod's algorithm counts
        # every selection as one event, regardless of whether
        # the sites end up interacting.
        if similarity == 0.0:
            # No features shared; cannot interact.
            return
        if similarity == 1.0:
            # Identical cultures; interaction happens but is a no-op.
            return
 
        # Bernoulli draw at probability = similarity
        if self.random.random() < similarity:
            # Identify features where the two cultures disagree.
            # Guaranteed non-empty because similarity < 1.0.
            differing = np.where(self.culture != neighbor.culture)[0]
            feature_idx = self.random.choice(differing)
            # The active site adopts the neighbor's trait on this feature.
            # This both reduces their cultural distance and
            # makes future interaction more likely, the positive feedback
            # that drives convergence within zones.
            self.culture[feature_idx] = neighbor.culture[feature_idx]
