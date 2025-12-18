# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState is None:
            return float('-inf')
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')
        score = successorGameState.getScore()
        capsules = successorGameState.getCapsules()
        food = newFood.asList()
        if food:
            min_food = min(manhattanDistance(newPos, f) for f in food)
            score += 1.0/(min_food+1)
        if capsules:
            min_cap = min(manhattanDistance(newPos, c) for c in capsules)
            score += 0.5/(min_cap+1)
        for g in newGhostStates:
            dist = manhattanDistance(newPos, g.getPosition())
            if g.scaredTimer > 0:
                score += 2.0 / (dist + 1)
            else:
                if dist == 0:
                    return float('inf')
                if dist <= 2:
                    score -= 3.0 / dist
        if action == Directions.STOP:
            score -= 0.2
        score -= 0.1*len(food)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        num_agents = gameState.getNumAgents()

        def minimax(state, depth, agent):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            legal = state.getLegalActions(agent)
            if not legal:
                return self.evaluationFunction(state), None

            def next_params(next_agent):
                next_depth = depth + 1 if next_agent == 0 else depth
                return next_depth, next_agent

            if agent == 0:  
                best_val, best_act = float('-inf'), None
                for a in legal:
                    s2 = state.generateSuccessor(agent, a)
                    v, _ = minimax(s2, *next_params((agent + 1) % num_agents))
                    if v > best_val:
                        best_val, best_act = v, a
                return best_val, best_act
            else: 
                best_val, best_act = float('inf'), None
                for a in legal:
                    s2 = state.generateSuccessor(agent, a)
                    v, _ = minimax(s2, *next_params((agent + 1) % num_agents))
                    if v < best_val:
                        best_val, best_act = v, a
                return best_val, best_act

        _, act = minimax(gameState, 0, 0)
        return act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        A, B = float('-inf'), float('inf')
        N = gameState.getNumAgents()

        def ab(state, depth, agent, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            legal = state.getLegalActions(agent)
            if not legal:
                return self.evaluationFunction(state), None

            def next_params(next_agent):
                next_depth = depth + 1 if next_agent == 0 else depth
                return next_depth, next_agent

            if agent == 0:  
                v, best = float('-inf'), None
                for a in legal:
                    s2 = state.generateSuccessor(agent, a)
                    nv, _ = ab(s2, *next_params((agent + 1) % N), alpha, beta)
                    if nv > v:
                        v, best = nv, a
                    if v > beta:      
                        return v, best
                    alpha = max(alpha, v)
                return v, best
            else:  
                v, best = float('inf'), None
                for a in legal:
                    s2 = state.generateSuccessor(agent, a)
                    nv, _ = ab(s2, *next_params((agent + 1) % N), alpha, beta)
                    if nv < v:
                        v, best = nv, a
                    if v < alpha:     
                        return v, best
                    beta = min(beta, v)
                return v, best

        _, action = ab(gameState, 0, 0, A, B)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        n = gameState.getNumAgents()
        def expval(state, depth, agent):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            legal = state.getLegalActions(agent)
            if not legal:
                return self.evaluationFunction(state), None
            def next_params(next_agent):
                next_depth = depth + 1 if next_agent == 0 else depth
                return next_depth, next_agent
            
            if agent == 0:
                best_val, best_act = float('-inf'), None
                for a in legal:
                    s2 = state.generateSuccessor(agent, a)
                    v, _ = expval(s2, *next_params((agent+1) % n))
                    if v > best_val:
                        best_val, best_act = v, a
                return best_val, best_act
            else:
                total = 0.0
                for a in legal:
                    s2 = state.generateSuccessor(agent, a)
                    v, _ = expval(s2, *next_params((agent+1) % n))
                    total += v
                return total/len(legal), None
        _, action = expval(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    caps = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    if food:
        dmin_food = min(manhattanDistance(pos, f) for f in food)
        score += 1.5 / (dmin_food + 1)
        score -= 2.0 * len(food)
    score -= 3.0 * len(caps)
    if caps:
        dmin_cap = min(manhattanDistance(pos, c) for c in caps)
        score += 0.5 / (dmin_cap + 1)
    for g in ghosts:
        d = manhattanDistance(pos, g.getPosition())
        if g.scaredTimer > 0:
            score += 4.0 / (d + 1)
        else:
            if d == 0:
                return float('-inf')
            if d <= 2:
                score -= 6.0 / d
            else:
                score -= 0.2 / (d + 1)
    return score

# Abbreviation
better = betterEvaluationFunction
