(* ::Package:: *)

BeginPackage["EccentricIMR`EccentricPN`"];

Needs/@{"EccentricIMR`EccentricPNSymbols`", "EccentricIMR`GopuGihyuk3PN`"};



EccentricSoln;
EccentricPNSolution;
xeModel;
$EccentricPNWaveformOrder;
$EccentricPNComputePsi4;
EccProfileCallTime;
EccProfileCallCount;
EccProfileReportTime;
ClearEccProfileCall;
EccProfileReportCount;

Begin["`Private`"];

Col[i_, l_] := Map[{#[[1]], #[[i]]} &, l];

AmpPhase[tb_List] :=
  Module[{ampPhaseTb,z,t,previousPhase, i, currentPhase = 0, cycles =
          0, nPoints},
  nPoints = Length[tb];
  ampPhaseTb = Table[i, {i, 1, nPoints}];
  For[i = 1, i <= nPoints, i++,
   z = tb[[i, 2]];
   t = tb[[i, 1]];
   currentPhase = ArcTan[Re[z], Im[z]];
   If[currentPhase - previousPhase > Pi, cycles--];
   If[currentPhase - previousPhase < -Pi, cycles++];
   previousPhase = currentPhase;
   ampPhaseTb[[i]] = {t, Abs[z], 2 Pi cycles + currentPhase}];
  Return[ampPhaseTb]]

Phase[tb_List] := Col[3, AmpPhase[tb]];

Amplitude[tb_List] := Col[2, AmpPhase[tb]];

(* Truncate a power series to its first n terms *)
FirstTerms[SeriesData_[var_, about_, terms_,nmin_,nmax_,den_], n_] :=
  Module[{coeffs,powers},
    coeffs = Take[terms,n];
    powers = Take[Table[var^(i/den), {i, nmin, nmax}],n];
    Dot[coeffs,powers]];


ClearEccProfileCall[] :=
  Module[{},
    Clear[EccProfileCallTime];
    Clear[EccProfileCallCount]];

SetAttributes[RecordProfile, HoldAll];

RecordProfile[name_, code_] := (*code;*)
   Module[{time, result, name2, result2, subTimers},
    name2 = Evaluate[name];
    {time, result} = AbsoluteTiming[ReleaseHold[code]];
    If[Head[EccProfileCallTime[name2]] === EccProfileCallTime, EccProfileCallTime[name2] = 0.0];
    If[Head[EccProfileCallCount[name2]] === EccProfileCallCount, EccProfileCallCount[name2] = 0];
    EccProfileCallTime[name2] += time;
    EccProfileCallCount[name2] += 1;
    result];

EccProfileReportTime[] :=
  Module[{},
    Reverse@SortBy[DownValues[EccProfileCallTime], #[[2]] &] /. 
    HoldPattern -> HoldForm /. ((x_ :> y_) :> {y, x}) /. 
    EccProfileCallTime[x_] :> x // TableForm];

EccProfileReportCount[] :=
  Module[{},
    Reverse@SortBy[DownValues[EccProfileCallCount], #[[2]] &] /. 
    HoldPattern -> HoldForm /. ((x_ :> y_) :> {y, x}) /. 
    EccProfileCallCount[x_] :> x // TableForm];

pnSolnToAssoc[soln_List, q_] :=
  Module[{},
    Block[{$ContextPath = Prepend[$ContextPath, "EccentricIMR`EccentricPNSymbols`"]},
      Join[Association@Table[ToString[var[[1]]] -> var[[2]], {var,soln}],Association["q"->q]]]];

etaOfq[q_] :=
 q/(1 + q)^2;

$cacheDir = FileNameJoin[{FileNameDrop[FindFile["EccentricIMR`EccentricPN`"],-1],
  "ExpressionCache"}];

SetAttributes[cacheResult, HoldAll];
cacheResult[filep_,expr_] :=
  If[file === None,
    expr,
    With[{file = FileNameJoin[{$cacheDir, filep}]},
      If[FileExistsQ[file], Block[{$ContextPath = Prepend[$ContextPath, "EccentricIMR`EccentricPNSymbols`"]}, (*Print["Using cached ", filep]; *)Get[file]], 
        (* else *)
        With[{result = expr},
          (* Print["Computed ", filep];  *)
          If[!DirectoryQ[FileNameDrop[file,-1]],
            CreateDirectory[FileNameDrop[file,-1],CreateIntermediateDirectories->True]];
          Module[{tempFile = file<>"."<>ToString[$KernelID]<>".tmp"},
            Put[result, tempFile];
            RenameFile[tempFile, file]];
          result]]]];

nInxe =  x^(3/2) - (3 x^(5/2))/(
 1 - e^2) + ((-18 + 28 eta + e^2 (-51 + 26 eta)) x^(7/2))/(
 4 (1 - e^2)^2) - (1/(
 128 (1 - e^2)^(
  7/2)))(16 e^4 (-240 + 
       156 Sqrt[1 - e^2] + (96 - 110 Sqrt[1 - e^2]) eta + 
       65 Sqrt[1 - e^2] eta^2) + 
    e^2 (96 (20 + 87 Sqrt[1 - e^2]) + 5120 Sqrt[1 - e^2] eta^2 + 
       3 eta (-256 - 7040 Sqrt[1 - e^2] + 
          41 Sqrt[1 - e^2] \[Pi]^2)) + 
    4 (-48 (-10 + Sqrt[1 - e^2]) + 224 Sqrt[1 - e^2] eta^2 + 
       eta (-192 - 3656 Sqrt[1 - e^2] + 
          123 Sqrt[1 - e^2] \[Pi]^2))) x^(9/2) + (1/(
 221184 (1 - e^2)^(
  9/2)))(256 e^6 (-1080 (-99 + 7 Sqrt[1 - e^2]) + 
       27 (-3712 + 71 Sqrt[1 - e^2]) eta - 
       945 (-32 + 5 Sqrt[1 - e^2]) eta^2 + 
       6422 Sqrt[1 - e^2] eta^3) + 
    3 e^4 (-86400 (-320 + 183 Sqrt[1 - e^2]) + 
       8018944 Sqrt[1 - e^2] eta^3 + 
       eta (-44654592 + 61957632 Sqrt[1 - e^2] + 188928 \[Pi]^2 - 
          656649 Sqrt[1 - e^2] \[Pi]^2) + 
       288 eta^2 (29952 - 149552 Sqrt[1 - e^2] + 
          1107 Sqrt[1 - e^2] \[Pi]^2)) + 
    8 (51840 (-40 + 13 Sqrt[1 - e^2]) + 100352 Sqrt[1 - e^2] eta^3 + 
       288 eta^2 (-3648 - 34504 Sqrt[1 - e^2] + 
          1353 Sqrt[1 - e^2] \[Pi]^2) + 
       3 eta (1855488 - 3114368 Sqrt[1 - e^2] - 11808 \[Pi]^2 + 
          80331 Sqrt[1 - e^2] \[Pi]^2)) + 
    12 e^2 (-13824 (565 + 112 Sqrt[1 - e^2]) + 
       1707008 Sqrt[1 - e^2] eta^3 + 
       3456 eta^2 (-608 - 8262 Sqrt[1 - e^2] + 
          205 Sqrt[1 - e^2] \[Pi]^2) - 
       3 eta (-3196928 - 12527744 Sqrt[1 - e^2] + 7872 \[Pi]^2 + 
          460527 Sqrt[1 - e^2] \[Pi]^2))) x^(11/2)      +  (x^(3/2) ((2 ((1 - e^2) (1 - (56374811 e^2)/24380301 + (103729937 e^4)/
           57112735 - (105413189 e^6)/194334558 + (49804512 e^8)/
           1158420851 + (4447985 e^10)/4076572203) ((263415291 e)/
           819998 - (3032926060 e^3)/3359177 - (974131935 e^5)/
           4216762 + (2317329404 e^7)/2491579 - (3829962720 e^9)/
           13601521 - (40653054 e^11)/124087307 + (199674118 e^13)/
           328008227) + (-1 + e^2) (-((112749622 e)/24380301) + (
           414919748 e^3)/57112735 - (105413189 e^5)/32389093 + (
           398436096 e^7)/1158420851 + (44479850 e^9)/
           4076572203) (286746937/12927762 + (263415291 e^2)/
           1639996 - (758231515 e^4)/3359177 - (324710645 e^6)/
           8433524 + (579332351 e^8)/4983158 - (382996272 e^10)/
           13601521 - (6775509 e^12)/248174614 + (14262437 e^14)/
           328008227) + 
        7 e (1 - (56374811 e^2)/24380301 + (103729937 e^4)/
           57112735 - (105413189 e^6)/194334558 + (49804512 e^8)/
           1158420851 + (4447985 e^10)/4076572203) (286746937/
           12927762 + (263415291 e^2)/1639996 - (758231515 e^4)/
           3359177 - (324710645 e^6)/8433524 + (579332351 e^8)/
           4983158 - (382996272 e^10)/13601521 - (6775509 e^12)/
           248174614 + (14262437 e^14)/328008227)) eta x^4)/(5 e (-1 +
         e^2)^4 (1 - (56374811 e^2)/24380301 + (103729937 e^4)/
        57112735 - (105413189 e^6)/194334558 + (49804512 e^8)/
        1158420851 + (4447985 e^10)/4076572203)^2) - ((1209 e^6 + 
      8 e^2 (3637 - 2917 Sqrt[1 - e^2]) + 
      4 e^4 (5982 - 2803 Sqrt[1 - e^2]) - 
      1152 (-1 + Sqrt[1 - e^2])) eta x^4)/(90 e^2 (-1 + e^2)^4) + (
   2 (1256 + 1608 e^2 + 111 e^4) eta x^4 Log[(
     4 (1 - e^2) (*(1 - Sqrt[1 - e^2])*) E^EulerGamma Sqrt[x])(*/e^2*)((1/2-(9 e^2)/8+(7 e^4)/8-(35 e^6)/128+(15 e^8)/512-e^10/2048)/(1-(5 e^2)/2+(9 e^4)/4-(7 e^6)/8+(35 e^8)/256-(3 e^10)/512))]  )/(
   15 (-1 + e^2)^4)))       (*This last parenthsis has the non-local part*) ;


h22 = (-4*eta*Sqrt[Pi/5]*(1 + r*(phiDot*r + I*rDot)^2))/(E^((2*I)*phi)*r*R);

etaFixed = $eta; 

konigsh22 = h22 /. {eta -> $eta, R -> 1};

h22InOrbital0PN = (-4*eta*Sqrt[Pi/5]*(1 + r*(phiDot*r + I*rDot)^2))/(E^((2*I)*phi)*r*R);
h22InOrbital1PN = (1/(7 r^2 R))E^(-2 I phi) eta Sqrt[Pi/5] (210 - 11 phiDot^2 r^3 + 
   3 (-7 eta - 52 eta phiDot^2 r^3 + 9 (-1 + 3 eta) phiDot^4 r^6) + 
   2 I phiDot r^2 (-5 (5 + 27 eta) + 
      27 (-1 + 3 eta) phiDot^2 r^3) rDot + 3 (15 + 32 eta) r rDot^2 + 
   54 I (-1 + 3 eta) phiDot r^3 rDot^3 + 27 (1 - 3 eta) r^2 rDot^4);

h22InOrbital = {h22InOrbital0PN,0,2/3 h22InOrbital1PN} /. {eta -> $eta, R -> 1};
hAmpPhaseExpr = hamp[t] Exp[I hphase[t]];
psi4AmpPhaseExpr = cacheResult["psi4AmpPhaseExpr.m", Simplify[D[hAmpPhaseExpr, t, t]]];
psi4DotAmpPhaseExpr = cacheResult["psi4DotAmpPhaseExpr.m", Simplify[D[hAmpPhaseExpr, t, t, t]]];


xeModel = {
  X -> x,
  Y -> e,
  X0 -> x0,
  Y0 -> e0,
  XDot -> xDotInxe  /. {x -> x[t], e -> e[t]},
  YDot -> eDotInxe /. {x -> x[t], e -> e[t]},
  nInXY -> Normal[nInxe] /. eta -> etaFixed
  };

EccentricSoln[args___] :=
  (Print["Invalid arguments: ", HoldForm[EccentricSoln[args]]];
    StringJoin[ToString[#,InputForm]&/@Riffle[{args},","],"]"];Abort[]);

EccentricPNSolution[params_Association, {t1_, t2_, dt_:1.0}] :=
  Module[{pnSolnRules},
    pnSolnRules = EccentricSoln[xeModel, N@etaOfq[params["q"]],
      {N@params["x0"], N@params["e0"] /. (0. -> 10.^-15), N@params["l0"], N@params["phi0"]},
      N@params["t0"], N/@{t1, t2, dt}];
    pnSolnToAssoc[pnSolnRules, params["q"]]];

EccentricSoln[model1_, eta0_?NumberQ, {x0_Real, y0_Real, l0_Real, phi0_},
              t0_?NumberQ, {t1p_?NumberQ, t2p_?NumberQ, 
              dt_?NumberQ}] :=
  Module[{x,y, xySoln, xFn, yFn, xTb,yTb,
          tTb, ephTb, betaphTb, eph0, betaPhi0, lSoln, lFn, lTb, uTb,  uFn, rTb, MikkolaCorrec, phiDootTb,
          vmuTb, WphTb,WphFn,omTb,   \[Lambda]phTb, \[Lambda]Fn,
          rFn, rDotFn, rDotTb, omFn, phiDootFn, phiSoln, phiFn, phiTb,
          ord = 5, t2, delta = 3*dt, indeterminate, extend, t1, return, 
          adiabatic, lEqs, model},
    
    RecordProfile["Setup",
    model = model1 /.$eta->eta0;
    x = (X/.model);
    y = (Y/.model);
    indeterminate = {_ -> Indeterminate};
    extend = !(phi0 === None);

    t1 = If[extend, Min[t0,t1p], t1p];
    t2 = t1 + Ceiling[t2p-t1,dt];

    If[t0 < t1 || t0 > t2,
      Print["EccentricSoln: ERROR: reference time t0 = ", t0, " is outside the solution domain ", {t1,t2}];
      Abort[]];

    If[x0 < 0 || Abs[y0] > 1,
      Print["WARNING: Invalid parameters ", {x0,y0,l0,phi0}, "; returning indeterminate"];
      Return[indeterminate]];

    return = False;
    adiabatic = {D[x[t], t] == Re[(XDot /. model)], 
                        D[y[t], t] == Re[(YDot /. model)], 
         x[t0] == N@x0, y[t0] == N@y0} /. eta -> N@eta0     /. x00 -> N@x0   ];

    RecordProfile["Adiabatic ODE solve",

    Quiet[xySoln = NDSolve[adiabatic,
      {x, y}, {t, Floor[Min[t1,t0]-delta,dt], Ceiling[t2+delta,dt]},StepMonitor:>(Global`$EccentricState=t)][[1]],NDSolve::ndsz]];
    Module[{rhs0 = adiabatic[[All,2]] //. {t->t0, x[t0]->N@x0, y[t0]->N@y0}},
      If[!And@@Map[NumberQ, rhs0],
        Print["Initial adiabatic ODE RHS is non-numeric: ", rhs0];
        Abort[]]];

    RecordProfile["Intermediate",

    t2 = t1 + Floor[(x/.xySoln)[[1,1,2]]-t1,dt];
    xFn = x /. xySoln; yFn = y /. xySoln;
 
    If[xFn[[1]][[1]][[2]] < t1, Print["Eccentric PN solution finished at ", xFn[[1]][[1]][[2]], " which is before the requested start time ", t1]; Return[indeterminate]]];

    RecordProfile["Tabulate x, y and t",

    xTb = Table[xFn[t], {t, t1, t2, dt}];
    yTb = Table[yFn[t], {t, t1, t2, dt}];
    tTb = Table[t, {t, t1, t2, dt}]];

(*RecordProfile["Compute betaphTb",
    betaphTb = betaphIneph   /. {x -> xTb, y -> yTb ,eta->  N@eta0}      ];*)
    
    If[Length[xTb] < 2, 

      Return[{psi4Om -> Indeterminate, psi4Phi -> Indeterminate,
        hOm -> Indeterminate}]];

    RecordProfile["l solve",
    lEqs = {D[l[t], t] == ((nInXY/.model)/.{x->x[t],y->y[t]}), l[t0] == l0} /. xySoln;
    lSoln = Quiet[NDSolve[lEqs, {l}, 
      {t, Min[t1,t0], Max[t2,t0]}][[1]],NDSolve::ndsz]];

    lFn = l /. lSoln;
    RecordProfile["Tabulate l",
    lTb = Map[lFn, tTb]];
                                   
uTb = MikkolaList[lTb,yTb,xTb]  ;

Do[ 
uTb[[i]] = MikkolaCorrection[  yTb[[i]] ,   xTb[[i]]  ,   uTb[[i]]  ,eta0 ];  
, {i,  Length[lTb]}];




WphTb = Wph    /. {x -> xTb, y -> yTb ,eta->  N@eta0, u->uTb }       ;
WphFn = Interpolation[MapThread[List, {tTb, WphTb}], InterpolationOrder->ord];

RecordProfile    ["Compute uFn",
    uFn = Interpolation[MapThread[List, {tTb, uTb}], InterpolationOrder->ord]];
    
    RecordProfile["Compute rTb",
    rTb = rr /. {x -> xTb, y -> yTb ,eta->  N@eta0, u->uTb }  ]    ;
    
    RecordProfile["Tabulate rDot",
    rDotTb =  rDoot /. {x -> xTb, y -> yTb ,eta->  N@eta0, u->uTb }  ]    ;
    
    omTb = xTb^(3/2);  

    RecordProfile["Interpolate omTb",
    omFn = Interpolation[MapThread[List, {tTb, omTb}], InterpolationOrder->ord]];  
          
   \[Lambda]phTb = \[Lambda]ph    /. {x -> xTb, y -> yTb ,eta->  N@eta0, u->uTb , l -> lTb }       ;
   \[Lambda]Fn = Interpolation[MapThread[List, {tTb, \[Lambda]phTb}], InterpolationOrder->ord];
          phiFn  = \[Lambda]Fn  +  WphFn   ;
           
       RecordProfile["Tabulate phi",
    phiSoln=NDSolve[{D[phi[t],t]==xFn[t]^(3/2),phi[0]==0},phi,{t,t1,t2}][[1]];
    phiFn=phi/.phiSoln;
    phiTb=Table[phiFn[t]+WphFn[t],{t,t1,t2,dt}];
    phiFn=phiFn+WphFn;];
    phiDootTb = phiDoot    /. {x -> xTb, y -> yTb ,eta->  N@eta0, u->uTb }   ;
   
    RecordProfile["Interpolate omTb2",
    phiDootFn = Interpolation[MapThread[List, {tTb, phiDootTb}], InterpolationOrder->ord]];
 
    RecordProfile["Rest",
    coords = {r -> rFn, om -> phiDootFn, phi -> phiFn};
    vars = {x -> xFn, y -> yFn, u -> uFn, l -> lFn};];
    waveform = EccentricWaveform[eta0, tTb, phiTb, rTb, rDotTb, phiDootTb, ord];

    Return[Join[coords, vars, waveform]];
];


MikkolaList[lTb_List,yTb_List,xTb_List] :=
  Module[{uTemp,uTb,i},  

uTb = Range[Length[lTb]];

Do[ 
uTb[[i]] = Mikkola[  lTb[[i]] ,   yTb[[i]]   ];
, {i,  Length[lTb]}];

Return[uTb];

];
Mikkola[time2_, e_] :=
  Module[  {ND,SIGN,SIGN2, alpha, beta1, zplus, zminus, z, s,w,E0,u,u1,u2,u3,u4,xi,sol,time},
  
If[  time2 < 0,  ND = Ceiling[time2/(2*Pi)] ; ,    ND=Floor[time2/(2*Pi)]; ];

SIGN = MikkolaSign[time2];
time = SIGN time2;

If[time > (2*Pi),     time = (time - Floor[time/(2.0*Pi)]*2*Pi);   ]  ;
If[time < Pi,SIGN2=1; ,   SIGN2=-1;   ] ;
If[SIGN2==1, ,    time=(2*Pi - time); ]   ;

 alpha  =  (1.0-e)/(4.0*e + 1/2.0);
 beta1  = (time/2.0)/(4.0*e + 1/2.0);
 zplus  = Power[beta1 + Sqrt[Power[alpha,3.0] + Power[beta1,2]]  ,(1.0/3.0)];
 zminus = Power[beta1 - Sqrt[Power[alpha,3.0] + Power[beta1,2]]  ,(1.0/3.0)];

 If[MikkolaSign[beta1] > 0 , z=zplus   ,  z=zminus ] ;
 z=zplus;
 s = (z - alpha/z);
 w = (s - (0.078*Power[s,5])/(1.0 + e));
 E0 = (time + e*(3*w - 4*Power[w,3]));
 u = E0 ;
 
 u1 = -Mikkolaf[u,e, time]/Mikkolaf1[u,e];
 u2 = -Mikkolaf[u,e,time]/(Mikkolaf1[u,e] + (1.0/2.0)*Mikkolaf2[u,e]*u1);
 u3 = -Mikkolaf[u,e,time]/(Mikkolaf1[u,e] + (1.0/2.0)*Mikkolaf2[u,e]*u2 + (1/6.0)*Mikkolaf3[u,e]*(Power[u2,2]));
 u4 = -Mikkolaf[u,e,time]/(Mikkolaf1[u,e] + (1.0/2.0)*Mikkolaf2[u,e]*u3 + (1/6.0)*Mikkolaf3[u,e]*(Power[u3,2]) + (1/24.0)*Mikkolaf4[u,e]*(Power[u3,3]));
 xi = (E0 + u4);
 
If[SIGN2>0,    sol=xi     ,         sol=(2.0*Pi - xi)      ]      ;
If[SIGN2>0,    sol= -sol     ,        u     = (sol + ND*2.0*Pi)           ]      ;

Return[u];

];

Mikkolaf[u_,e_,time_]:= 
Module[  {var1},

var1= u - e Sin[u]- time;
Return[  var1];
]

Mikkolaf1[u_,e_]:= 
Module[  {var1},
var1= 1. - e Cos[u];
Return[  var1];
]

Mikkolaf2[u_,e_]:= 
Module[  {var1},
var1=  e Sin[u];
Return[  var1];
]

Mikkolaf3[u_,e_]:= 
Module[  {var1},
var1=  e Cos[u];
Return[  var1];
]

Mikkolaf4[u_,e_]:= 
Module[  {var1},
var1=  -e Sin[u];
Return[  var1];
]
MikkolaSign[x_]:= 
Module[  {var1},
If[x<0, 
var1 = -1,
var1 = 1
];
Return[var1];
]


EccentricWaveform[eta_, tTb_List, phiTb_List, rTb_List, rDotTb_List, phiDootTb_List,
                  ord_Integer] :=
  Module[{t1, t2, dt, hTb, hFn, hPhase, hPhaseFn, hOmFn, hAmp, hAmpFn, psi4Tb,
          psi4Fn, psi4Phase, psi4PhaseFn, psi4DotTb, psi4OmTb, psi4OmFn, order,
         h22Expr},

    t1 = First[tTb];
    t2 = Last[tTb];
    If[Length[tTb] < 2,
      Print["EccentricWaveform: Input data too short (tTb = "<>ToString[tTb]<>")"];
      Abort[]];
    dt = tTb[[2]] - tTb[[1]];

    order = If[!ValueQ[$EccentricPNWaveformOrder], 0, $EccentricPNWaveformOrder];

    h22Expr = Plus@@Take[h22InOrbital /. $eta->eta, 2 order + 1];
    RecordProfile["Evaluate h22",
    hTb = h22Expr /. {phi -> phiTb, r -> rTb, rDot -> rDotTb, 
         phiDot -> phiDootTb}];

    RecordProfile["Interpolate h",
    hFn = Interpolation[MapThread[List, {tTb, hTb}], InterpolationOrder->ord]];
    RecordProfile["Evaluate h phase",
    hPhase = Phase[MapThread[List, {tTb, hTb}]];
    hPhaseFn = Interpolation[hPhase, InterpolationOrder->ord]];
    hOmFn = Derivative[1][hPhaseFn];
    
    RecordProfile["Compute psi4",
    If[$EccentricPNComputePsi4 =!= False,
      hAmp = Amplitude[MapThread[List, {tTb, hTb}]];
      hAmpFn = Interpolation[hAmp, InterpolationOrder->ord];
      psi4Tb =
      Table[psi4AmpPhaseExpr /. {hamp -> hAmpFn, hphase -> hPhaseFn,t -> tx}, {tx, t1, t2,
        dt}];

      psi4Fn = Interpolation[MapThread[List, {tTb, psi4Tb}], InterpolationOrder->ord];
      psi4Phase = Phase[MapThread[List, {tTb, psi4Tb}]];
      psi4PhaseFn = Interpolation[psi4Phase, InterpolationOrder->ord];

      psi4DotTb =
      Table[psi4DotAmpPhaseExpr /. {hamp -> hAmpFn, hphase -> hPhaseFn,
        t -> tx}, {tx, t1, t2, dt}];
      psi4OmTb = MapThread[Im[#1/#2] &, {psi4DotTb, psi4Tb}];
      psi4OmFn = Interpolation[MapThread[List, {tTb, psi4OmTb}],
        InterpolationOrder->ord]]];

    Return[{h -> hFn, hPhi -> hPhaseFn, psi4 -> psi4Fn, psi4Om -> psi4OmFn, 
            psi4Phi -> psi4PhaseFn, hOm -> hOmFn}];
  ];


End[];

EndPackage[];
