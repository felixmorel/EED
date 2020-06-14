# Economic Emission Dispatch

This README file explains how to obtain all figures from the master's thesis:
"Economic Emission Dispatch" from Félix Morel de Westgaver.

The codes are written in Python, here is a list of all files that are used:

Domain.py, DynamicD.py, Fixed_E_or_F.py, Grad.py, LinApproxLVH.py, MultiPOZ.py, NLSolverStat.py, NonConvexLoss.py, NonQuadratic.py, Params.py, POZ.py, RelaxComparison.py, Relaxed.py, Simple_Hermite.py, Static_model.py


In order to get the figures and tables from the thesis, install first Gurobi (version used 9.0.1) for Python and get a license from https://www.gurobi.com/.
Do the following in Python:


## Chapter 1

RUN Static_model.py

Figure 1.3.1: call	Simple(6, Spacing=0),
			Simple(10,Spacing=0),
			Simple(40,Spacing=0)
			

Figure 1.3.2: call 	Simple(6, tan='yes')


Figure 1.3.3: call	Simple(6, Spacing=2),
			Simple(10,Spacing=2),
			Simple(40,Spacing=2)
			

Figure 1.3.4: call	Simple(6),
			Simple(10),
			Simple(40)
			
RUN Fixed_E_or_F.py

Figure 1.4.1: call 	eConst(6, 'E')


Figure 1.5.1: call 	eConst(6, 'C')


RUN Simple_Hermite.py

Figure 1.6.1: call	main(6),
			main(10),
			main(40),
			

Figure 1.6.2: call	main(6,True)

## Chapter 2

RUN DynamicD.py

Figure 2.1.1: call	figures(6)


Figure 2.2.1: call	figures(10)

## Chapter 3

RUN NonQuadratic.py

Figure 3.1.1: call 	figures(10)

## Chapter 4

RUN NonConvexLoss.py

Figure 4.2.1: call 	figures(10)


Figure 4.2.2: close all figures and call
			figures(6),
			figures(6,'LimE'),
			figures(6,'LimF')

RUN Grad.py	

Figure 4.3.1: call	Gradmethod(10,'NonConvex') (if wanted change parameter h=2/(mu+L))


Figure 4.3.2: call	Accmethod(10,'NonConvex')

RUN Domain.py

Figure 4.4.1,
Figure 4.4.2,
Figure 4.5.2: call	figures(2)


Figure 4.4.3,
Figure 4.5.1: call 	figures(3)


RUN RelaxComparison.py

Table 4.1: call		TestNegdes(2), TestNegdes(3), TestNegdes(6), TestNegdes(10)


Figure 4.7.1,
Figure 4.7.3: call	ErrorRelaxation()


Figure 4.7.4,
Figure 4.7.5: call	TimeRelaxation()


RUN Grad.py

Figure 4.8.1: call	Gradmethod(10),
			Accmethod(10)


Figure 4.8.2: call	SPG(10),
			SPG(40)


RUN NLSolverStat.py

Figure 4.9.1: call	testSQP(100)

## Chapter 5

RUN POZ.py

Figure 5.2.1,
Figure 5.2.2:  call	figures()


RUN MultiPOZ

Figure 5.2.3: call	multiPOZ()


Figure 5.3.1: call	ErrorPOZ()


