import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)

    # return (forward_path, forward_prob)
    path = []
    p_prob = 1.0
    SymbolSets = ['-'] + SymbolSets
    for t in range(y_probs.shape[1]):
        p_t = y_probs[:,t,:]
        best_next = np.argmax(p_t)
        path.append(SymbolSets[best_next])
        p_prob = p_prob * p_t[best_next][0]
    
    s = ''
    for i in range(len(path)):
        c = path[i]
        if i == 0:
            s += c
            prev_c = c
            continue
        if c == prev_c:
            continue
        if c == '-':
            prev_c = c
            continue
        s += c
        prev_c = c
        
    return s, p_prob



##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
def BeamSearch(SymbolSets, y_probs, BeamWidth):  
    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0,0])
    # Subsequent time steps
    for t in range(1, y_probs.shape[1]):
        # Prune the collection down to the BeamWidth
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)
        # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(BlankPathScore, PathScore, PathsWithTerminalBlank,PathsWithTerminalSymbol, y_probs[:,t,0])
        # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(BlankPathScore, PathScore, PathsWithTerminalBlank,PathsWithTerminalSymbol, SymbolSets, y_probs[:,t,0])
    # Merge identical paths differing only by the final blank
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
    # Pick best path
    BestScore = max(FinalPathScore.values())
    for k, v in FinalPathScore.items():
        if v == BestScore:
            return k, FinalPathScore

def InitializePaths(SymbolSet, y):
    InitialBlankPathScore = {}
    InitialPathScore = {}
    
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = ''
    InitialBlankPathScore[path] = y[0] # Score of blank at t=1
    InitialPathsWithFinalBlank = [path]
    
    # Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathsWithFinalSymbol = []
    for i in range(len(SymbolSet)): # This is the entire symbol set, without the blank
        path = SymbolSet[i]
        InitialPathScore[path] = y[i+1] # Score of symbol c at t=1
        InitialPathsWithFinalSymbol.append(path) # Set addition
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol,InitialBlankPathScore, InitialPathScore


def ExtendWithBlank(BlankPathScore, PathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
    UpdatedPathsWithTerminalBlank = []
    UpdatedBlankPathScore = {}
    # First work on paths with terminal blanks
    #(This represents transitions along horizontal trellis edges for blanks)
    for path in PathsWithTerminalBlank:
        # Repeating a blank doesn’t change the symbol sequence
        UpdatedPathsWithTerminalBlank.append(path) # Set addition
        UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]

    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
        # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path]* y[0]
        else:
            UpdatedPathsWithTerminalBlank.append(path) # Set addition
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank,UpdatedBlankPathScore


def ExtendWithSymbol(BlankPathScore, PathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
    UpdatedPathsWithTerminalSymbol = []
    UpdatedPathScore = {}
    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSet)): # SymbolSet does not include blanks
            newpath = path + SymbolSet[i] # Concatenation
            UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]

    # Next work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
        # Extend the path with every symbol other than blank
        for i in range(len(SymbolSet)): # SymbolSet does not include blanks
            c = SymbolSet[i]
            if c != path[-1]:
                newpath = path + c
            else:
                newpath = path
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    score_list = list(BlankPathScore.values()) + list(PathScore.values())
    score_list.sort(reverse=True)
    cutoff = score_list[min(BeamWidth, len(score_list))-1]
    PrunedPathsWithTerminalBlank = []
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] >= cutoff:
            PrunedPathsWithTerminalBlank.append(p)# Set addition
            PrunedBlankPathScore[p] = BlankPathScore[p]

    PrunedPathsWithTerminalSymbol = []
    for p in PathsWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathsWithTerminalSymbol.append(p)# Set addition
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    # All paths with terminal symbols will remain
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.append(p) # Set addition
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore