/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2014 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kOmegaSSTx_clustering.H"
#include "addToRunTimeSelectionTable.H"

#include "backwardsCompatibilityWallFunctions.H"

#include <iostream>
#include <fstream>
using namespace std;
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
namespace RASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(kOmegaSSTx_clustering, 0);
addToRunTimeSelectionTable(RASModel, kOmegaSSTx_clustering, dictionary);

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

tmp<volScalarField> kOmegaSSTx_clustering::F1(const volScalarField& CDkOmega) const
{
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar("1.0e-10", dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*nu()/(sqr(y_)*omega_)
            ),
            (4*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


tmp<volScalarField> kOmegaSSTx_clustering::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*nu()/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}


tmp<volScalarField> kOmegaSSTx_clustering::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*nu()/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


tmp<volScalarField> kOmegaSSTx_clustering::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23() *= F3();
    }

    return f23;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kOmegaSSTx_clustering::kOmegaSSTx_clustering
(
    const volVectorField& U,
    const surfaceScalarField& phi,
    transportModel& transport,
    const word& turbulenceModelName,
    const word& modelName
)
:
    RASModel(modelName, U, phi, transport, turbulenceModelName),

    alphaK1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK1",
            coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK2",
            coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega1",
            coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega2",
            coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma1",
            coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma2",
            coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta1",
            coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta2",
            coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "a1",
            coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "b1",
            coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "c1",
            coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::lookupOrAddToDict
        (
            "F3",
            coeffDict_,
            false
        )
    ),

    y_(mesh_),

    k_
    (
        IOobject
        (
            "k",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateK("k", mesh_)
    ),
    omega_
    (
        IOobject
        (
            "omega",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateOmega("omega", mesh_)
    ),
    nut_
    (
        IOobject
        (
            "nut",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateNut("nut", mesh_)
    ),
    p_
    (
        IOobject
        (
            "p",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_ //autoCreateP("p", mesh_)
    ),
    Rstress_
    (
        IOobject
        (
            "R", //turbulenceProperties:R
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(U_))/omega_
    ),
    Ax_
    (
        IOobject
        (
            "Ax",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(U_))/omega_
    ),
    Rx_
    (
        IOobject
        (
            "Rx",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(U_))/omega_
    )
{
    bound(k_, kMin_);
    bound(omega_, omegaMin_);

    nut_ =
    (
        a1_*k_
      / max
        (
            a1_*omega_,
            b1_*F23()*sqrt(2.0)*mag(symm(fvc::grad(U_)))
        )
    );
    nut_.correctBoundaryConditions();

    printCoeffs();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

tmp<volSymmTensorField> kOmegaSSTx_clustering::R() const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                "R",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            ((2.0/3.0)*I)*k_ - nut_*twoSymm(fvc::grad(U_)) + 2*k_*Ax_,
            k_.boundaryField().types()
        )
    );
}


tmp<volSymmTensorField> kOmegaSSTx_clustering::devReff() const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                "devRhoReff",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
           -nuEff()*dev(twoSymm(fvc::grad(U_))) + dev(2*k_*Ax_)
        )
    );
}


tmp<fvVectorMatrix> kOmegaSSTx_clustering::divDevReff(volVectorField& U) const
{
    return
    (
      - fvm::laplacian(nuEff(), U)
      - fvc::div(nuEff()*dev(T(fvc::grad(U))))
      + fvc::div(dev(2*k_*Ax_))
    );
}


tmp<fvVectorMatrix> kOmegaSSTx_clustering::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    volScalarField muEff("muEff", rho*nuEff());

    return
    (
      - fvm::laplacian(muEff, U)
      - fvc::div(muEff*dev(T(fvc::grad(U))))
      + fvc::div(dev(2*rho*k_*Ax_))
    );
}


bool kOmegaSSTx_clustering::read()
{
    if (RASModel::read())
    {
        alphaK1_.readIfPresent(coeffDict());
        alphaK2_.readIfPresent(coeffDict());
        alphaOmega1_.readIfPresent(coeffDict());
        alphaOmega2_.readIfPresent(coeffDict());
        gamma1_.readIfPresent(coeffDict());
        gamma2_.readIfPresent(coeffDict());
        beta1_.readIfPresent(coeffDict());
        beta2_.readIfPresent(coeffDict());
        betaStar_.readIfPresent(coeffDict());
        a1_.readIfPresent(coeffDict());
        b1_.readIfPresent(coeffDict());
        c1_.readIfPresent(coeffDict());
        F3_.readIfPresent("F3", coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


void kOmegaSSTx_clustering::correct()
{
    RASModel::correct();

    if (!turbulence_)
    {
        return;
    }

    if (mesh_.changing())
    {
        y_.correct();
    }

    // Calculate basis tensors and invariants
    volTensorField gradU = fvc::grad(U_);
    volSymmTensorField Sij = dev(symm(gradU));
    volTensorField Wij = -0.5*(gradU - gradU.T());

    volScalarField S = sqrt(2*magSqr(symm(gradU)));
    volScalarField tau = 1./max(S/a1_ + omegaMin_, omega_ + omegaMin_);                         // Check if limiter is necessary

    volSymmTensorField sij = Sij * tau;
    volTensorField wij = Wij * tau;

    volScalarField I01 = tr(sij & sij);
    volScalarField I02 = tr(wij & wij);
	volScalarField I03 = tr(sij & (sij & sij));
	volScalarField I04 = tr(wij & (wij & sij));
	volScalarField I05 = tr(wij & (wij & (sij & sij)));

    volSymmTensorField T01 = sij;
    volSymmTensorField T02 = symm((sij & wij) - (wij & sij));
    volSymmTensorField T03 = symm(sij & sij) - scalar(1.0/3.0)*I*I01;
    volSymmTensorField T04 = symm(wij & wij) - scalar(1.0/3.0)*I*I02;
	volSymmTensorField T05 = symm((wij & (sij & sij)) - ((sij & sij) & wij));
	volSymmTensorField T06 = symm(((wij & wij) & sij) + ((sij & sij) & wij) - (2.0/3.0) * I * tr(sij & (wij & wij)));
	volSymmTensorField T07 = symm((wij & (sij & (wij & wij))) - (wij & (wij & (sij & wij))));
	volSymmTensorField T08 = symm((sij & (wij & (sij & sij))) - (sij & (sij & (wij & sij))));
	volSymmTensorField T09 = symm((wij & (wij & (sij & sij))) + (sij & (sij & (wij & wij))) - (2.0/3.0) * I * tr(sij & (sij & (wij & wij))));
	volSymmTensorField T10 = symm((wij & (sij & (sij & (wij & wij)))) -(wij & (wij & (sij & (sij & wij)))));

    dimensionedScalar kSMALL("0",dimLength*dimLength/dimTime/dimTime, 1e-10);
    dimensionedScalar rSMALL ("0", dimensionSet(0,2,-2,0,0,0,0),1e-10);
    dimensionedScalar USMALL ("0", dimensionSet(0,1,-1,0,0,0,0),1e-10);
    dimensionedScalar qSMALL ("0", dimensionSet(0,0,-1,0,0,0,0),1e-10);
    dimensionedScalar NewSMALL("0", dimensionSet(0,0,0,0,0,0,0), 1e-10);
    dimensionedScalar constantInRe("0", dimensionSet(0,0,0,0,0,0,0), 2);
    wallDist y(mesh_);

    // Q1  
    volScalarField Q1_org =  0.5*(Foam::sqr(tr(gradU)) - tr(((gradU) & (gradU))));
    volScalarField Q1 (
        IOobject (
            "Q1",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ
        ),
        (Q1_org-Foam::min(Q1_org))/(Foam::max(Q1_org)-Foam::min(Q1_org)),
        "zeroGradient"
    );
    // Info << "--> Q1_org Min:" << Foam::min(Q1_org).value() << " Max: " << Foam::max(Q1_org).value() << endl;
    // Info << "--> Q1     Min:" << Foam::min(Q1).value() << " Max: " << Foam::max(Q1).value() << endl;

    // Q2
    volScalarField Q2 (
            IOobject (
                    "Q2",
                    runTime_.timeName(),
                    mesh_,
                    IOobject::NO_READ
            ),
            k_/ (0.5* (U_&U_) + k_ + kSMALL),
            "zeroGradient"
    );
    // Info << "--> Q2    Min:" << Foam::min(Q2).value() << " Max :" << Foam::max(Q2).value() << endl;


    Info<< "    Reading transport Properties" <<endl;
    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            runTime_.constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );

    dimensionedScalar nu (transportProperties.lookup("nu"));
    Info << "--> nu  :" << nu.value() << endl;

    // Q3
    volScalarField Q3 (
        IOobject (
            "Q3",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ
        ),
        mag(y * Foam::sqrt(k_+kSMALL) / nu)/10000,
        "zeroGradient"
    );
    // Info << "--> Q3    Min:" << Foam::min(Q3).value() << " Max :" << Foam::max(Q3).value() << endl;

    // pressure field
    // Info<< "    Reading the pressure" << endl;
    //IOobject Pheader (
    //        "p",
    //        runTime_.timeName(),
    //        mesh_,
    //        IOobject::AUTO_WRITE,
    //        IOobject::READ_IF_PRESENT
    //);
    //volScalarField p(Pheader, mesh_);

    // Info << "--> Min P:" << Foam::min(p).value() << endl;
    // Info << "--> Max P:" << Foam::max(p).value() << endl;

    const volScalarField& p_ = mesh_.objectRegistry::template lookupObject<volScalarField>("p");
    volVectorField gradp = fvc::grad(p_);
    // Info << "--> Min gradp:" << Foam::min(gradp).value() << endl;
    // Info << "--> Max gradp:" << Foam::max(gradp).value() << endl;

    volScalarField Q4_org = U_ & gradp;
    volScalarField Q4 (
        IOobject (
            "Q4",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ
        ),
        (Q4_org-Foam::min(Q4_org))/(Foam::max(Q4_org)-Foam::min(Q4_org)),
        "zeroGradient"
    );
    //Info << "--> Q4_org Min:" << Foam::min(Q4_org).value() << " Max: " << Foam::max(Q4_org).value() << endl;
    //Info << "--> Min Q4:" << Foam::min(Q4).value() << "Max :" << Foam::max(Q4).value() << endl;
    
    // Q5
    volScalarField Q5_org = mag(fvc::curl(U_));
    volScalarField Q5 (
            IOobject (
                "Q5",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ
            ),
            (Q5_org-Foam::min(Q5_org))/(Foam::max(Q5_org)-Foam::min(Q5_org)),
            "zeroGradient"
        );
        //Info << "--> Q5_org Min:" << Foam::min(Q5_org).value() << " Max: " << Foam::max(Q5_org).value() << endl;
        //Info << "--> Min Q5:" << Foam::min(Q5).value() <<  "Max :" << Foam::max(Q5).value() << endl;

    // Q6
    volScalarField Q6 (
    IOobject (
                "Q6",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ
            ),
            nut_ / (1. * nu*100 + nut_),
            "zeroGradient"
    );
    //Info << "--> Q6    Min:" << Foam::min(Q6).value() << " Max :" << Foam::max(Q6).value() << endl;


    // Q7
    volScalarField Q7_org = Foam::sqrt(gradp & gradp);
    volScalarField Q7 (
    IOobject (
                "Q7",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ
            ),
            (Q7_org-Foam::min(Q7_org))/(Foam::max(Q7_org)-Foam::min(Q7_org)),
            "zeroGradient"
    );
    //Info << "--> Q7_org Min:" << Foam::min(Q7_org).value() << " Max: " << Foam::max(Q7_org).value() << endl;
    //Info << "--> Min/Max Q7:" << Foam::min(Q7).value() << Foam::max(Q7).value() << endl;

    // Q8
    volScalarField Q8_org = mag((U_ * U_) && gradU);
        volScalarField Q8 (
            IOobject (
                "Q8",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ
            ),
            (Q8_org-Foam::min(Q8_org))/(Foam::max(Q8_org)-Foam::min(Q8_org)),
            "zeroGradient"
     );
     // Info << "--> Q8_org Min:" << Foam::min(Q8_org).value() << " Max: " << Foam::max(Q8_org).value() << endl;
     // Info << "--> Min/Max Q8:" << Foam::min(Q8).value() <<  Foam::max(Q8).value() << endl;

    // Q9
    volScalarField Q9_org = (fvc::grad(k_) && U_);
    volScalarField Q9 (
            IOobject (
                "Q9",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ
            ),
            (Q9_org-Foam::min(Q9_org))/(Foam::max(Q9_org)-Foam::min(Q9_org)),
           "zeroGradient"
    );
    //Info << "--> Q9_org Min:" << Foam::min(Q9_org).value() << " Max: " << Foam::max(Q9_org).value() << endl;
    //Info << "--> Min/Max Q9:" << Foam::min(Q9).value() << Foam::max(Q9).value() << endl;

    Info<< "    Reading the Reynolds stress : R" <<endl;
//    IOobject Rheader (
//            "R",
//            runTime_.timeName(),
//            mesh_,
//            IOobject::MUST_READ
//    );
//    volSymmTensorField R(Rheader, mesh_);


//    Rstress_(
//            IOobject (
//            "turbulenceProperties:R",
//            runTime_.timeName(),
//            mesh_,
//            IOobject::MUST_READ,
//            IOobject::NO_WRITE
//            ),
//            mesh_
//    );

    // Q10
    //tmp<volTensorField> tgradU = fvc::grad(U_);
    //const volTensorField& gradU_ = tgradU();  //using const-ref object, not the tmp func.
    //Rstress_= ((2.0/3.0)*I)*k_ - this->nut_*dev(twoSymm(tgradU()))+ Ax_;
    //Rstress_= this->R();
    volScalarField Q10 (
    IOobject (
                "Q10",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ
            ),
            //Foam::sqrt(R && R)/(k_ + kSMALL + Foam::sqrt(R && R)),
            //Foam::sqrt(Rstress_ && Rstress_)/(k_ + kSMALL + Foam::sqrt(Rstress_ && Rstress_)),
            Foam::sqrt(this->R() && this->R())/(k_ + kSMALL + Foam::sqrt(this->R() && this->R())),
            "zeroGradient"
    );
    // Info << "--> Q10    Min:" << Foam::min(Q10).value() << " Max :" << Foam::max(Q10).value() << endl;
    
    // Get trained turbulence model corrections
    #include "nonLinearModel.H"

    // Set additional anisotropy to zero for low k
    forAll(Ax_, idx)
    {
	    if (k_[idx] <= 1e-7)
	    {
	        Ax_[idx] = (Ax_[idx]-Ax_[idx]);
	    }
    }

    // not really elegant to put here ...
    int num_cells = this->mesh_.cells().size();
    int ClusterId[num_cells];
    //Info << " num_cells = " << num_cells << endl;

    if (this->runTime_.time().timeOutputValue() == 10001) // very bad!
    {
        forAll(Ax_,id)
        {
            ClusterId[id] = 9999;
        }
        Info << " ClusterId initialized " << endl;

        Info << "Reading labels file ..." << endl;
        ifstream file;

        file.open("labels.dat");

        string line;
        int i = 0;

        while(file.good() && (getline(file, line)))
        {
            ClusterId[i] = stod(line);
            i++;
        }

        file.close();
    }

     // use clusterId to assign GEP models
      forAll(Ax_,celli)
      {

        if (ClusterId[celli] == 0) {
            //  Expression: 0   ((I1-Q5)+(((((Q8)*(((0.0)*(-1.0)))))+(0.0326458107713309))-I2))
            //  Expression: 1   ((-0.44498745107701737-I1)+(Q10+0.0))
            //  Expression: 2   ((((Q3)-(1.0))*0.0)+(1.0+0.0))
            //  Expression: 3   ((-21.68933360479937*Q10)+(11.626587684561484+Q2))
            //  Expression: 4   ((0.0-((((((Q9)+(Q10)))+(Q2)))-(((0.9657165809719962)-(Q7)))))+(0.0*0.0))
            //  Expression: 5   ((-0.10239668670259565-0.0)+(0.0+Q5))
            //  Expression: 6   ((0.0+Q2)+(-1.375283991275608+((I1)-(((((0.0)-(Q3)))-(Q7))))))
            //  Expression: 7   ((0.0+30.34625402412766)+(0.0-((I1)*(293.7937020807219))))
            //  Expression: 8   ((Q2-Q4)+(0.0+Q1))
            //  Expression: 9   ((6.3530141130354405+0.0)+(0.0-0.0))
            Ax_[celli] = (( ((I01[celli]-Q5[celli])+(((((Q8[celli])*(((0.0)*(-1.0)))))+(0.0326458107713309))-I02[celli])) * T01[celli] ) +
                         ( ((-0.44498745107701737-I01[celli])+(Q10[celli]+0.0)) * T02[celli] ) + 
                         ( ((((Q3[celli])-(1.0))*0.0)+(1.0+0.0)) * T03[celli] ) +
                         ( ((-21.68933360479937*Q10[celli])+(11.626587684561484+Q2[celli])) * T04[celli] ) +
                         ( ((0.0-((((((Q9[celli])+(Q10[celli])))+(Q2[celli])))-(((0.9657165809719962)-(Q7[celli])))))+(0.0*0.0)) * T05[celli] ) +
                         ( ((-0.10239668670259565-0.0)+(0.0+Q5[celli])) * T06[celli] ) +
                         ( ((0.0+Q2[celli])+(-1.375283991275608+((I01[celli])-(((((0.0)-(Q3[celli])))-(Q7[celli])))))) * T07[celli] ) +
                         ( ((0.0+30.34625402412766)+(0.0-((I01[celli])*(293.7937020807219)))) * T08[celli] ) +
                         ( ((Q2[celli]-Q4[celli])+(0.0+Q1[celli])) * T09[celli] ) +
                         ( ((6.3530141130354405+0.0)+(0.0-0.0)) * T10[celli] ) );

            //Ax_[celli] = (T01[celli]-T01[celli]); 
            Rx_[celli] = (T01[celli]-T01[celli]);
        }
        else if (ClusterId[celli] == 1) {
             //After 1000 generations, best colony has index 64 and fitness value of 0.005003
             //    Expression: 0   ((-16.765560899730506-0.0)*(((0.6701242508952023)-(((Q9)-(((-12.424401785425923)*(Q8))))))+I1))
             //    Expression: 1   ((Q10-Q3)+(-0.6834172734040134+((((Q8)*(248.55608918617685)))*(8.266179316239116))))
             //    Expression: 2   ((0.0+-1.3886499852923915)+(Q6*((((((73.4266318654393)*(Q10)))-(37.81511393059079)))+(Q6))))
             //    Expression: 3   ((Q9*-1.8281696761235584)+(0.0+0.0))
             //    Expression: 4   ((0.0*0.0)-(0.0+0.0))
             //    Expression: 5   ((0.0*0.0)-(0.0--0.3306432765234347))
             //    Expression: 6   ((0.0+0.0)+(0.0+0.0))
             //    Expression: 7   ((0.0+0.0)-(0.0*0.0))
             //    Expression: 8   ((Q6+-0.436128951393892)*(-595.4318562933066*-595.4318562933066))
             //    Expression: 9   ((0.0+0.0)*(0.0+((Q9)*(I3))))
            Ax_[celli] = (( ((-16.765560899730506-0.0)*(((0.6701242508952023)-(((Q9[celli])-(((-12.424401785425923)*(Q8[celli]))))))+I01[celli])) * T01[celli] ) +
                         ( ((Q10[celli]-Q3[celli])+(-0.6834172734040134+((((Q8[celli])*(248.55608918617685)))*(8.266179316239116)))) * T02[celli] ) + 
                         ( ((0.0+-1.3886499852923915)+(Q6[celli]*((((((73.4266318654393)*(Q10[celli])))-(37.81511393059079)))+(Q6[celli])))) * T03[celli] ) + 
                         (  ((Q9[celli]*-1.8281696761235584)+(0.0+0.0)) * T04[celli] ) +
                         ( ((0.0*0.0)-(0.0+0.0)) * T05[celli] ) +
                         ( ((0.0*0.0)-(0.0+0.3306432765234347)) * T06[celli] ) + 
                         ( ((0.0+0.0)+(0.0+0.0)) * T07[celli] ) +
                         ( ((0.0+0.0)-(0.0*0.0)) * T08[celli] ) +
                         ( ((Q6[celli]+-0.436128951393892)*(-595.4318562933066*-595.4318562933066)) * T09[celli] ) +
                         ( ((0.0+0.0)*(0.0+((Q9[celli])*(I03[celli])))) * T10[celli] ) );
            //Ax_[celli] = (T01[celli]-T01[celli]);
            Rx_[celli] = (T01[celli]-T01[celli]);
        }
        else if (ClusterId[celli] == 2) { 
            // Best colony has index 86 and fitness value of 0.006478
            //     Expression: 0   ((-0.10767340984881715+((I1)*(-8.66071614466871)))+(Q3+0.0))
            //     Expression: 1   ((((0.46904084342649277)+(Q8))+Q7)-(Q10+((Q6)*(-0.04562092752555417))))
            //     Expression: 2   ((0.0+Q2)+(0.0*0.0))
            //     Expression: 3   ((((Q10)-(0.8995279185198728))-0.0)+(Q10*Q9))
            //     Expression: 4   ((0.0-0.0)*(0.0*0.0))
            //     Expression: 5   ((0.0+0.0)-(0.0+2.654350321053868))
            //     Expression: 6   ((0.0-0.0)+(-1.6514634799669887-I2))
            //     Expression: 7   ((19.360265954237857-0.0)*(Q10+((I2)+(-0.3285861806610472))))
            //     Expression: 8   ((2.0+-3.904375972859942)+(0.0*0.0))
            //     Expression: 9   ((Q1+0.0)+(0.0-0.0))
            //     Lowest complexity colony has index 2 and complexity of 44.000000
            //     Stats of entire population
            //     Fitness value sum:  0.6532552616291086
            //     Fitness value mean:     0.006532552616291086
            //     Fitness value std:  4.95274575400739e-05
            //     Evaluation time:    520.7486476898193
            Ax_[celli] = ( ( ((-0.10767340984881715+((I01[celli])*(-8.66071614466871)))+(Q3[celli]+0.0)) * T01[celli] ) + 
                           ( ((((0.46904084342649277)+(Q8[celli]))+Q7[celli])-(Q10[celli]+((Q6[celli])*(-0.04562092752555417)))) * T02[celli] ) + 
                           ( ((0.0+Q2[celli])+(0.0*0.0)) * T03[celli] ) + 
                           ( ((((Q10[celli])-(0.8995279185198728))-0.0)+(Q10[celli]*Q9[celli])) * T04[celli] ) + 
                           ( ((0.0-0.0)*(0.0*0.0)) * T05[celli] ) + 
                           (  ((0.0+0.0)-(0.0+2.654350321053868)) * T06[celli] ) + 
                           ( ((0.0-0.0)+(-1.6514634799669887-I02[celli])) * T07[celli] ) + 
                           ( ((19.360265954237857-0.0)*(Q10[celli]+((I02[celli])+(-0.3285861806610472)))) * T08[celli] ) + 
                           ( ((2.0+-3.904375972859942)+(0.0*0.0)) * T09[celli] ) +
                           ( ((Q1[celli]+0.0)+(0.0-0.0)) * T10[celli] ) );
            //Ax_[celli] = (T01[celli]-T01[celli]);
            Rx_[celli] = (T01[celli]-T01[celli]); 
        } 
        else if (ClusterId[celli] == 3) { 
            Ax_[celli] = (T01[celli]-T01[celli]);
            Rx_[celli] = (T01[celli]-T01[celli]); 
        }
        else if (ClusterId[celli] == 4) { 
            Ax_[celli] = (T01[celli]-T01[celli]);
            Rx_[celli] = (T01[celli]-T01[celli]); 
        }
        else if (ClusterId[celli] == 5) { 
            Ax_[celli] = (T01[celli]-T01[celli]);
            Rx_[celli] = (T01[celli]-T01[celli]); 
        }
        else if (ClusterId[celli] == 6) { 
            Ax_[celli] = (T01[celli]-T01[celli]);
            Rx_[celli] = (T01[celli]-T01[celli]); 
        }
        else // like for example 9999 as it was initialized
        {
            Ax_[celli] = (T01[celli]-T01[celli]); // baseline model
            Rx_[celli] = (T01[celli]-T01[celli]); // baseline model
        }
      }

    volScalarField S2(2*magSqr(symm(gradU)));
    volScalarField G(GName(), nut_*S2);
    
    volScalarField Gc = nut_*S2 - 2*k_ * (Ax_ && symm(gradU));
    volScalarField Rc = 2*k_ * (Rx_ && symm(gradU));

    dimensionedScalar min_nut
    (
        "min_nut",
        dimensionSet(0, 2, -1, 0, 0, 0 ,0),
        1e-25
    );

    // Update omega and G at the wall
    omega_.boundaryField().updateCoeffs();

    const volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    const volScalarField F1(this->F1(CDkOmega));

    // Turbulent frequency equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(omega_)
      + fvm::div(phi_, omega_)
      - fvm::laplacian(DomegaEff(F1), omega_)
     ==
        gamma(F1) * min(Gc /(nut_+min_nut), (c1_/a1_)*betaStar_*omega_*max(a1_*omega_, b1_*F23()*sqrt(S2)))
      + gamma(F1) * mag(Rc)/(nut_+min_nut)
      - fvm::Sp(beta(F1)*omega_, omega_)
      - fvm::SuSp
        (
            (F1 - scalar(1))*CDkOmega/omega_,
            omega_
        )
    );

    omegaEqn().relax();

    omegaEqn().boundaryManipulate(omega_.boundaryField());

    solve(omegaEqn);
    bound(omega_, omegaMin_);

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(k_)
      + fvm::div(phi_, k_)
      - fvm::laplacian(DkEff(F1), k_)
     ==
        min(Gc, c1_*betaStar_*k_*omega_)
      + mag(Rc)
      - fvm::Sp(betaStar_*omega_, k_)
    );

    kEqn().relax();
    solve(kEqn);
    bound(k_, kMin_);


    // Re-calculate viscosity
    nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));
    nut_.correctBoundaryConditions();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
