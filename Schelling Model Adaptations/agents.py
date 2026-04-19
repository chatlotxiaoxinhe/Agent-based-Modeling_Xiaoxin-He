from mesa import Agent

class SchellingAgent(Agent):
    ## Initiate agent instance, inherit model trait from parent class
    def __init__(self, model, agent_type):
        super().__init__(model)
        ## Set agent type
        self.type = agent_type
        # Agent inherits the global desired share as its baseline
        self.base_tolerance = self.model.tolerance
        # Current tolerance
        self.current_tolerance = self.base_tolerance
        # Tracker for consecutive moves without finding a satisfactory neighborhood
        self.consecutive_moves = 0

    ## Define basic decision rule
    def step(self):
        ## Get list of neighbors within range of sight
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        
        ## Count neighbors of same type as self
        similar_neighbors = sum(1 for neighbor in neighbors if neighbor.type == self.type)
       
        ## If an agent has any neighbors (to avoid division by zero), calculate share of neighbors of same type
        if len(neighbors) > 0:
            share_alike = similar_neighbors / len(neighbors)
        else:
            share_alike = 1.0

        # Bounded Rationality
        # Evaluate neighborhood against the agent's current tolerance
        if share_alike < self.current_tolerance:
            # If an agent is unhappy with neighbors, relocate to a random empty slot
            self.model.grid.move_to_empty(self)
            
            # Increment
            self.consecutive_moves += 1
            
            # If the agent reaches the relocation threshold, it reduces tolerance
            if self.consecutive_moves >= self.model.decay_threshold:
                self.current_tolerance = max(0.0, self.current_tolerance - self.model.decay_amount)
                # Reset the consecutive moves counter after adapting
                self.consecutive_moves = 0  
                
        else: 
            # Happy with current neighborhood, contribute to global happiness tracker
            self.model.happy += 1
            # Reset search friction counter since the agent is now settled
            self.consecutive_moves = 0