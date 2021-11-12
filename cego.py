import os
#os.environ['RPY2_CFFI_MODE'] = "API" # bug in cffi 1.13.0 https://bitbucket.org/rpy2/rpy2/issues/591/runtimeerror-found-a-situation-in-which-we

import numpy as np
import pandas as pd
from mallows_kendall import distance


from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
from rpy2.robjects import numpy2ri
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import STAP
numpy2ri.activate()
import rpy2.rinterface as ri

from r_problem import make_r_fitness
#from mallows_kendall import kendallTau
# # funcion de distancia entre permutaciones
# @ri.rternalize
# def r_kendallTau(A,B):
#     return kendallTau(A,B)

def cego(instance, seed, budget, m_ini, budgetGA, eval_ranks, dist_name):
    # Reset the list of recorded evaluations.
    instance.reset()
    
    rstring = """
    library(CEGO)
    print(sessionInfo())
    # This is identical to the function in the package but takes also a maxTime parameter.
    my_optimCEGO <- function (x = NULL, fun, control = list()) 
    {
    con <- list(evalInit = 2, vectorized = FALSE, verbosity = 0, 
        plotting = FALSE, targetY = -Inf, budget = 100, creationRetries = 100, 
        distanceFunction = distancePermutationHamming, creationFunction = solutionFunctionGeneratorPermutation(6), 
        infill = infillExpectedImprovement, model = modelKriging, 
        modelSettings = list(), optimizer = optimEA, optimizerSettings = list(), 
        initialDesign = designMaxMinDist, archiveModelInfo = NULL, 
        initialDesignSettings = list(), maxTime = 3600 * 24 * 6, eval_ranks = FALSE)
    con[names(control)] <- control
    control <- con
    rm(con)
    maxTime <- control$maxTime + proc.time()["elapsed"] 
    count <- control$evalInit
    archiveModelInfo <- control$archiveModelInfo
    vectorized <- control$vectorized
    verbosity <- control$verbosity
    plotting <- control$plotting
    creationFunction <- control$creationFunction
    distanceFunction <- control$distanceFunction
    if (is.null(control$initialDesignSettings$distanceFunction)) 
        control$initialDesignSettings$distanceFunction <- distanceFunction
    fun
    if (!vectorized) { 
        fn <- if (control$eval_ranks) function(x) unlist(lapply(x, fun)) 
              else function(x) unlist(lapply(x, function(y) fun(order(y))))
    } else {
     # fn <- fun
     stop("We do not handle vectorized functions")
    }
    res <- list(xbest = NA, ybest = NA, x = NA, y = NA, distances = NA, 
        modelArchive = NA, count = count, convergence = 0, message = "")
    msg <- "Termination message:"
    res$x <- control$initialDesign(x, creationFunction, count, 
        control$initialDesignSettings)
    distanceHasParam <- FALSE
    if (is.function(distanceFunction)) {
        if (length(distanceFunction) == 1) 
            distanceHasParam <- length(formalArgs(distanceFunction)) > 
                2
        else distanceHasParam <- any(sapply(sapply(distanceFunction, 
            formalArgs, simplify = FALSE), length) > 2)
        if (!distanceHasParam) 
            res$distances <- CEGO:::distanceMatrixWrapper(res$x, distanceFunction)
    }
    res$y <- fn(res$x)
    indbest <- which.min(res$y)
    res$ybest <- res$y[[indbest]]
    res$xbest <- res$x[[indbest]]
    model <- CEGO:::buildModel(res, distanceFunction, control)
    if (!is.null(archiveModelInfo)) {
        res$modelArchive <- list()
        archiveIndex <- 1
        if (identical(model, NA)) {
            res$modelArchive[[archiveIndex]] <- rep(NA, length(archiveModelInfo))
            names(res$modelArchive[[archiveIndex]]) <- archiveModelInfo
        }
        else {
            res$modelArchive[[archiveIndex]] <- model$fit[archiveModelInfo]
            names(res$modelArchive[[archiveIndex]]) <- archiveModelInfo
        }
    }
    useEI <- is.function(control$infill)
    while ((res$count < control$budget) & (res$ybest > control$targetY) & ((maxTime - proc.time()["elapsed"]) > 0)) {
        if (!identical(model, NA)) {
            optimres <- CEGO:::optimizeModel(res, creationFunction, 
                model, control)
            duplicate <- list(optimres$xbest) %in% res$x
            improved <- optimres$ybest < optimres$fpredbestKnownY
        }
        else {
            msg <- paste(msg, "Model building failed, optimization stopped prematurely.")
            warning("Model building failed in optimCEGO, optimization stopped prematurely.")
            res$convergence <- -1
            break
        }
        res$count <- res$count + 1
        if ((!duplicate && (improved || useEI)) || distanceHasParam) {
            res$x[[res$count]] <- optimres$xbest
        } else { # !distanceHasParam
                designSize <- length(res$x) + 1
                if (is.list(distanceFunction)) 
                  dfun <- distanceFunction[[1]]
                else dfun <- distanceFunction
                xc <- CEGO:::designMaxMinDist(res$x, creationFunction, 
                  designSize, control = list(budget = control$creationRetries, 
                    distanceFunction = dfun))
                res$x[[res$count]] <- xc[[designSize]]
                # Modified to print
                print(paste0("Random solution:", res$count))
        }
        res$x <- CEGO::removeDuplicates(res$x, creationFunction)
        res$y <- c(res$y, fn(res$x[res$count]))
        indbest <- which.min(res$y)
        res$ybest <- res$y[[indbest]]
        res$xbest <- res$x[[indbest]]
        if (verbosity > 0) {
            # Modified to print current quality
            print(paste("Evaluations:", res$count, "    Quality:", 
                res$y[res$count], "    Best:", res$ybest))
        }
        if (plotting) {
            plot(res$y, type = "l", xlab = "number of evaluations", 
                ylab = "y")
            abline(res$ybest, 0, lty = 2)
        }
        if (!distanceHasParam & is.function(distanceFunction)) 
            res$distances <- CEGO:::distanceMatrixUpdate(res$distances, 
                res$x, distanceFunction)
        model <- CEGO:::buildModel(res, distanceFunction, control)
        if (!is.null(archiveModelInfo)) {
            archiveIndex <- archiveIndex + 1
            if (identical(model, NA)) {
                res$modelArchive[[archiveIndex]] <- rep(NA, length(archiveModelInfo))
                names(res$modelArchive[[archiveIndex]]) <- archiveModelInfo
            }
            else {
                res$modelArchive[[archiveIndex]] <- model$fit[archiveModelInfo]
                names(res$modelArchive[[archiveIndex]]) <- archiveModelInfo
            }
        }
    }
    if (min(res$ybest, na.rm = TRUE) <= control$targetY) {
        msg <- paste(msg, "Successfully achieved target fitness.")
        res$convergence <- 1
    }
    else if (res$count >= control$budget) {
        msg <- paste(msg, "Target function evaluation budget depleted.")
    }
    else if ((maxTime - proc.time()["elapsed"]) <= 0) {
        msg <- paste(msg, "maxTime reached.")
    }
    res$message <- msg
    res$distances <- NULL
    res
    }

    my_cego <- function(fun, dist_name, n, m_ini = 5, budget = 15, seed = 0, budgetGA = 100, eval_ranks)
    {
    set.seed(seed)
    # mutation
    mF <- mutationPermutationInterchange
    # recombination
    rF <- recombinationPermutationCycleCrossover
    #creation
    cF <- function() sample(n)
    dist <-  switch(dist_name, 
                    kendall = distancePermutationSwap,
                    hamming = distancePermutationHamming,
                    dist_name)
    # start optimization
#    print("antes del optimCEGO")
    res <- my_optimCEGO(x = NULL,
                        fun = fun,
                        control = list(creationFunction=cF,
                                     distanceFunction = dist,
                                     optimizerSettings=list(budget=budgetGA,popsize=20,
                                                            mutationFunction=mF,
                                                            recombinationFunction=rF),
                                     evalInit=m_ini,budget=budget,verbosity=1,
                                     model=modelKriging,
                                     vectorized=FALSE, eval_ranks = eval_ranks))

    return(list(res$xbest, res$ybest, do.call(rbind, res$x), res$y))
    }
    """
    rcode = STAP(rstring, "rcode")
    # This function already converts from 1-based (R) to 0-based (Python)
    r_fitness = make_r_fitness(instance)
    best_x, best_fitness, x, y = rcode.my_cego(r_fitness,
                                               dist_name = dist_name,
                                               n = instance.n,
                                               m_ini = m_ini,
                                               budget = budget,
                                               seed = seed,
                                               budgetGA = budgetGA,
                                               eval_ranks = eval_ranks)

    # We use what instance recorded because CEGO may not get us what was
    # actually evaluated.
    return pd.DataFrame(dict(
        Fitness = instance.evaluations,
        x = [ ' '.join(map(str,s)) for s in instance.solutions ],
        m_ini = m_ini,
        seed = seed,
        budget = budget, budgetGA = budgetGA,
        eval_ranks = eval_ranks,
        Distance = [ instance.distance_to_best(perm, distance) for perm in instance.solutions]))
