import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import networkx as nx
import random

class Agent(nn.Module):
    def __init__(self, n_layers, n_nodes_per_layer, n_states, n_messages, n_in_links, batch_size):
        super(Agent, self).__init__()
        self.output = None  # Initialize output to None or a default tensor as necessary

        self.n_states = n_states # Number of variables representing the agent's state
        self.n_messages = n_messages # Number of variables representing incoming messages
        self.n_in_links = n_in_links # Number of incoming links from other agents/environments
        self.batch_size = batch_size # Number of samples in each batch

        # Calculate input and output dimensions
        input_dim = 1  # Placeholder, actual dimension will be set later
        output_dim = (n_states + n_messages) 

        # Create the neural network layers
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, n_nodes_per_layer))
            layers.append(nn.ReLU())
            current_dim = n_nodes_per_layer
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())  # Use Sigmoid to output binary variables
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def set_input_dimension(self, new_dim):
            # Adjust the input layer with the correct dimension
            self.network[0] = nn.Linear(new_dim, self.network[0].out_features)

    def forward(self, x):
        output = self.network(x.view(self.batch_size, -1))
        self.output = output  # Store the output for access by other agents
        # NOTE Currently assuming the output is a probability distribution. We should consider binarizing it.
        return output



class Environment:
    def __init__(self, num_samples, num_variables):
        """
        Initialize the environment with a specified number of samples and variables.
        :param num_samples: The number of samples in the environment.
        :param num_variables: The number of variables in this environment node.
        """
        # NOTE Currently assuming the nodes are not correlated. We should consider adding correlations to represent modularity.
        self.num_samples = num_samples
        self.num_variables = num_variables
        # Initialize all binary variables for each sample
        self.variables = torch.randint(0, 2, (num_samples, num_variables), dtype=torch.float32)
        # Initialize output
        self.output = self.variables.clone()  # Start with initial state as output

    def reset(self):
        """
        Resets the environment variables to a new random set of binary values.
        """
        self.variables = torch.randint(0, 2, (self.num_samples, self.num_variables), dtype=torch.float32)
        self.output = self.variables.clone()  # Reset output to new state

    def get_state(self):
        """
        Returns the current state of the environment.
        """
        return self.variables

    def set_state(self, new_state):
        """
        Updates the environment state with new binary variables.
        :param new_state: A matrix of new binary variables (0 or 1).
        """
        if new_state.shape != self.variables.shape:
            raise ValueError("The new state shape must match the current state shape.")
        self.variables = new_state.float()
        self.output = self.variables.clone()  # Update output to match new state



class Organization:
    def __init__(self, agents, environments, batch_size):
        """
        Initializes the Organization with a set of agents and environments.
        :param agents: A list of Agent instances.
        :param environments: A list of Environment instances.
        :param batch_size: The number of samples each agent and environment handles per batch.
        """
        self.agents = agents
        self.environments = environments
        self.graph = nx.DiGraph()  # Directed graph
        self.batch_size = batch_size

        # Add agents and environments to the graph
        for agent in agents:
            self.graph.add_node(agent, type='agent')
        for environment in environments:
            self.graph.add_node(environment, type='environment')

        # Randomly pick one agent as the Actor
        self.actor = random.choice(agents)
        # Calculate the total number of variables in all environments
        total_env_vars = sum(env.num_variables for env in environments) 
        # Adjust Actor's output dimension to match the total environment variables
        self.actor.network[-2] = nn.Linear(self.actor.network[-2].in_features, total_env_vars)
        self.actor.network[-1] = nn.Sigmoid()  # Assuming binary output is still desired
        # Initialize a single optimizer for all agents
        all_parameters = torch.nn.ParameterList()
        for agent in agents:
            all_parameters.extend(list(agent.network.parameters()))
        self.optimizer = optim.Adam(all_parameters, lr=0.001)

    def connect(self, from_node, to_node):
        """
        Connects two nodes in the organization's graph.
        """
        if from_node in self.graph and to_node in self.graph:
            self.graph.add_edge(from_node, to_node)
        else:
            raise ValueError("Both nodes must be part of the graph.")

    def update_agent_links(self):
        """
        Updates the n_in_links parameter of each agent based on the incoming edges in the graph.
        """
        for agent in self.agents:
            in_links = self.graph.in_degree(agent)
            agent.n_in_links = in_links

    def update_agent_input_dimensions(self):
            """
            Updates the input dimensions of each agent based on the actual inputs from their predecessors in the graph.
            """
            for agent in self.agents:
                total_input_dim = agent.n_states+sum(predecessor.output.shape[1] for predecessor in self.graph.predecessors(agent))
                agent.set_input_dimension(total_input_dim)


    def perform_message_passing(self):
        """
        Performs message passing between agents in the organization.
        Each agent receives a concatenated input of its own state and outputs from its predecessors.
        """
        # Initialize message inputs for each agent
        for agent in self.agents:
            #NOTE Currently assuming the order of the message passing is based on the agent number. We should consider randomizing it.
            # Start with the agent's own states
            state_input = torch.randn(self.batch_size, agent.n_states)  # Assuming state is stored or generated here

            # Collect outputs from predecessors
            message_inputs = []
            for predecessor in self.graph.predecessors(agent):
                if predecessor.output is not None:
                    message_inputs.append(predecessor.output)

            # Concatenate the agent's state with inputs from predecessors
            if message_inputs:
                total_inputs = torch.cat([state_input] + message_inputs, dim=1)
            else:
                total_inputs = state_input

            # Process the combined inputs through the agent's network
            # NOTE: Currently assuming the agents can output continuous values. We should consider binarizing it.
            agent.output = agent.forward(total_inputs)

    def calculate_objective(self):
        """
        Calculate the KL divergence between the concatenated environment states and the Actor's output.
        """
        # Concatenate all environment states
        env_states = torch.cat([env.get_state().reshape(self.batch_size, -1) for env in self.environments], dim=1)
        # Assume the Actor's output and environment states are probabilities
        p = dist.Categorical(env_states)
        q = dist.Categorical(self.actor.output)
        kl_div = dist.kl_divergence(p, q).mean()  # Average KL divergence over the batch
        return kl_div

    def train(self, num_message_passing):
        """
        Train the organization by performing message passing and optimizing the objective.
        """
        for _ in range(num_message_passing):
            self.perform_message_passing()
            kl_div = self.calculate_objective()
            self.optimizer.zero_grad()
            kl_div.backward()
            self.optimizer.step()
        return kl_div



if __name__ == '__main__':
    # Create Agents and Environments
    batch_size = 10
    agents = [Agent(n_layers=2, n_nodes_per_layer=10, n_states=5, n_messages=2, n_in_links=1, batch_size=batch_size) for _ in range(3)]
    environments = [Environment(num_samples=batch_size, num_variables=5) for _ in range(2)]

    # Initialize the Organization
    org = Organization(agents, environments, batch_size)

    # Connect the agents and environments in the graph
    org.connect(agents[0], environments[0])  # Agent 0 observes Environment 0
    org.connect(agents[1], environments[1])  # Agent 1 observes Environment 1
    org.connect(environments[0], agents[2])  # Environment 0 informs the Actor
    org.connect(environments[1], agents[2])  # Environment 1 informs the Actor

    # Update the in-links for agents based on the current graph
    org.update_agent_links()
    org.update_agent_input_dimensions()
    # Perform message passing
    org.perform_message_passing()

    # Train the organization
    final_kl_divergence = org.train(num_message_passing=10)
    print("KL Divergence after a step of training:", final_kl_divergence)