from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        '''

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            '''
            loads = []
            # TODO create all load ground actions from the domain Load action
            for p in self.planes:
                for c in self.cargos:
                    for a in self.airports:
                            precond_pos = [expr("At({}, {})".format(c, a)),
                                           expr("At({}, {})".format(p, a)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("In({}, {})".format(c, p))]
                            effect_rem = [expr("At({}, {})".format(c, a))]
                            load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            loads.append(load)            
            
            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            for p in self.planes:
                for c in self.cargos:
                    for a in self.airports:
                            precond_pos = [expr("In({}, {})".format(c, p)),
                                           expr("At({}, {})".format(p, a)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(c, a))]
                            effect_rem = [expr("In({}, {})".format(c, p))]
                            unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            unloads.append(unload) 
            
            return unloads

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
            flys = []
            for fr in self.airports:
                for p in self.planes:
                    for to in self.airports:
                        if fr != to:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []
        
        # Instantiate knowledge base class to hold state for processing.
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        # Loop over actions.
        # If preconditions are met, then add to list of possible.
        for action in self.actions_list:
            # Assume action is valid unless state clauses wrong
            clause_test_pos_pass=True
            clause_test_neg_pass=True
            
            # Below based on code snippet from aimacode.planning.Action class
            # check for positive clauses
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                     clause_test_pos_pass=False
                     break
            # check for negative clauses
            for clause in action.precond_neg:
                if clause in kb.clauses:
                     clause_test_neg_pass=False
                     break
            # if both positive and negative clauses appear as expected
            # add action to list.
            if clause_test_pos_pass and clause_test_neg_pass:
                 possible_actions.append(action)
                 
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        new_state = FluentState([], [])
               
        # Error trap for actions that are not part of action(state)
        legal_action_test=False
        for legalAction in self.actions(state):
             if action.name==legalAction.name and action.args==legalAction.args:
                  legal_action_test=True
             
        # If legal_action_continue else return unchanged state
        if not legal_action_test:
             return encode_state(new_state, self.state_map) 
        #
        # Update state based on action
        # 
        # Below based on code snippet from aimacode.planning.Action class 
        #
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        # check if the preconditions are satisfied
        # --preconditions are tested in the legal_action_test above.
        # remove negative literals
        for clause in action.effect_rem:
            kb.retract(clause)
        # add positive literals
        for clause in action.effect_add:
            kb.tell(clause)

        # update state object
        new_state.pos = kb.clauses
        
        # return result
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        #
        # From the Russell-Norvig book, the two basic steps
        # for this potentially inadmissible heuristic are
        #
        # (1) Relax actions by removing all preconditions and all effects
        # unless an effect is a literal in a goal.
        # (2) Count the minimum number of actions required such that the
        # union of effects satisfies the goal
        #
        # The second step is an NP-hard set cover problem, but the book
        # says there is a simple greedy algorithm which solves it within
        # a factor of log(n), n=number of literals in goal.
        #
        
        # Start of implementation
        #
        # Loop over all the actions and test to see if at least one of the
        # effects is in the goal.  If so, add the action to the 
        # running set.
        
        # determine state at current passed in node
        fs = decode_state(node.state, self.state_map)
        # create a holder for actions that can produce final state from here.
        actionSet = set()
        remainingGoals = set()
        # Loop over goal states and compare to action effects.
        for iGoal in self.goal:
            # If a goal has already been met by current state, skip to next goal.
            if iGoal in fs.pos:
                continue
            else:
                # add goal to target set
                remainingGoals.add(iGoal)
            for cur_action in self.actions_list:
                # get the list of positive literal effects
                pos_effects = cur_action.effect_add
                #if the goal is covered by the effects of this action, add the action
                if iGoal in pos_effects:
                    actionSet.add(cur_action)
        
        
        #
        # Apply a greedy set cover algorithm on the running action set
        # to see what the minimum set of actions are that achieve the 
        # remaining goal state literals that have not been met.
        #
        # For this portion, I implemented based on the description
        # from Wikipedia:  https://en.wikipedia.org/wiki/Set_cover_problem
        # for the greedy algorithm which chooses, at each step choose
        # the set which contains the largest number of uncovered
        # elements.
        #
        # In terms of the present problem, the "Universe" is
        # the set of remaining goal literals to be satisfied: remainingGoals
        # and the underlying "Collection" is the set, each element
        # of which is an action that produces at least one of the 
        # goal literals: actionSet
        #
        # The steps are to loop over actionSet, and select the
        # action that covers the largest number of remaining goals.
        # The effects are removed from the remaining goals set.
        # Once the remaining goals set is empty
        # terminate the algorithm and return the size of the covering set.
        #
        coveringSet = set()
        while len(remainingGoals)>0:
            # define variables to hold max effect actions at each round.
            max_action=None
            max_eff_count=0
            for iAction in actionSet:  #List of possible actions
                goal_eff_count=0
                # Loop over positive effects of current action
                for cur_eff in iAction.effect_add:
                    if cur_eff in remainingGoals:
                        goal_eff_count = goal_eff_count+1
                # Keep track of action with most coverage
                if goal_eff_count > max_eff_count:
                    max_eff_count=goal_eff_count
                    max_action=iAction
             
            #Add maximum covering action to covering set
            coveringSet.add(max_action)
            # remove the effects from the remaining goals set
            for eff_to_remove in max_action.effect_add:
                remainingGoals.discard(eff_to_remove) #removes if present
            
        # return the size of the covering set
        return len(coveringSet)


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    neg = [expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C3, SFO)'),
           expr('At(C3, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),        
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P3, SFO)'),
           expr('At(P3, JFK)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),
           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),
           expr('At(C3, SFO)'),
           expr('At(C3, JFK)'),
           expr('At(C3, ORD)'),
           expr('At(C4, SFO)'),
           expr('At(C4, JFK)'),
           expr('At(C4, ATL)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C4, P1)'),
           expr('In(C4, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
