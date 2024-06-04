#include <stdio.h>
#include <unistd.h> // for usleep function
#include <stdlib.h>
#include <string.h> // for memset
#include <math.h>
#include <time.h> // to generate seed for prg
#include <locale.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include "gnuplot_i.h"


#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

///gcc sourcefile.c -o programname -lgsl -lgslcblas -lm -Wall
/// many improvements here to do but for the task its fine.
/// if you want to analyse the output pipe it to a txt file (eg. ./x > check.dat)



int Estimationloop = 128, episodes = 1000;  // TODO make just arguments for function    


gsl_rng *r;

double k[4]; // make global to ease manipulation of vector with functions
double avgEndtime=0.0, avgReward = 0.0; // variables to save our results
//  k1-4 | -€ 0 € | aend arew
double kEval[4][3][2];
double currentGradEnd[4]; //save our current evaluated gradient for each parameter k
double currentGradReward[4];



static const double g = 9.81, m_c= 5.0, m_p=2.0, l=0.8, controlInterval=0.01; 

typedef struct{
    double position;	// 2D position of cart pole , 1 DOF , y-fixed, so only x
    double xspeed;		// speed of cart in x direction
    double angle;		// angle btw. optimal equilibrium position and current postion of pole (theta)
    double angleVelocity;	
    double Force;
    double xspeedAccel;
    double angleAccel;
    int    currentReward;
    int    time;
} CartPoleState; 


////////////////////////// x dot dot

double calc_xspeedAccel(CartPoleState *currentState) {
	
	double tmpAccelaration;

	tmpAccelaration =  (currentState->Force - m_p * l * ( currentState->angleAccel * cos(currentState->angle) - ( pow(currentState->angleVelocity,2) * sin(currentState->angle) ) )) / (m_c + m_p );
	
	currentState->xspeedAccel = tmpAccelaration ;
	//printf("xspeedAccel: %f ",tmpAccelaration);

	return tmpAccelaration;

}

//////////////////////// theta dot dot

double calc_angleAccel(CartPoleState *currentState) {

	double tmpAccelaration;

	tmpAccelaration = g * sin(currentState->angle) * (m_c + m_p) - (  ( currentState->Force  + m_p * l * pow(currentState->angleVelocity,2) * sin(currentState->angleVelocity) ) * cos (currentState->angleVelocity) ) / ( 4/3 * l * (m_c + m_p) - (m_p * l * pow(cos(currentState->angleVelocity),2))) ;


	currentState->angleAccel = tmpAccelaration ;	
	//printf("angleAccel: %f ",tmpAccelaration);

	return tmpAccelaration;

}

//////////////////////

void updateSystemState(CartPoleState *currentState) {

	calc_angleAccel(currentState);		// don´t change order we need first the accel. of the angle to compute the x accel.
	calc_xspeedAccel(currentState);
	
	/// complete update struct -> transition to next time step, calculate the gaussian noise with variance for each component


	currentState->position = ( currentState->position + ( controlInterval * currentState->xspeed ) ) + gsl_ran_gaussian(r,0.001) ;
	currentState->xspeed   = ( currentState->xspeed + ( controlInterval * currentState->xspeedAccel ) ) + gsl_ran_gaussian(r,0.01); 
	currentState->angle    = ( currentState->angle + ( controlInterval * currentState->angleVelocity )  ) + gsl_ran_gaussian(r,0.001);
	currentState->angleVelocity = (currentState->angleVelocity + ( controlInterval * currentState->angleAccel) ) + gsl_ran_gaussian(r,0.01);
	

}

///////////////////////

double randomNumber(void){ // returns +/- floating point random numbers in interval ~ -20 <-> +20 with mean 0
		
	int scaler = 2000000000;			
	if (gsl_rng_uniform(r) < 0.5 ) return ( gsl_rng_uniform(r) ) * -1.0 * gsl_rng_get(r)/ gsl_rng_uniform_int(r,scaler);
		else return ( gsl_rng_uniform(r)  ) * gsl_rng_get(r)/ gsl_rng_uniform_int(r,scaler);


}

/////////////////// K Values

void initRandomKVector(double *V, int Lenght) {
	int i;	

	for (i = 0; i < Lenght; i++) *V++=randomNumber();

}



//////////////// K Values applied

void applyFeedbackForce(CartPoleState *currentState) {
	
	currentState->Force = MIN(100, MAX( -100, (k[0] * currentState->position + k[1] * currentState->xspeed + k[2] * currentState->angle + k[3] * currentState->angleVelocity)));

}

///////////////////

void initSystemState(CartPoleState *currentState) {

	// CartPoleState tmpState = {1.0,-0.5,-0.2,0.5,0.0,0.0,0.0,0};
	
	 currentState->position= 1.0;currentState->xspeed=-0.5;currentState->angle=-0.2;currentState->angleVelocity=0.5;currentState->Force=0.0;
	 currentState->xspeedAccel=0.0;currentState->angleAccel=0.0;currentState->currentReward=0;currentState->time=0;

}

////////////////

void calcFinalReward(CartPoleState *currentState) {

	currentState->currentReward+=- ( 1000 - currentState->time );

}



/////////////////////////////

int calcRunningReward(CartPoleState *currentState) {

	if  ( (abs(currentState->position) > 5 ) || (abs(currentState->angle) > 0.1 ) ) { calcFinalReward(currentState); return 1;}

	if  ( (abs(currentState->position) < 0.1 ) && (abs(currentState->angle) < 0.1 ) ) return 0;
		else currentState->currentReward-=1;
	
	return 0;

}

////// writes feedback into globals avgEndtime and avgReward


void gradEval(CartPoleState *currentState, double *parmK, double epsilon) {

avgEndtime=avgReward = 0.0; // reset variables to save our results for the next round

int i,j;
	
*parmK+=epsilon; // change parameter k with value epsiolon

  for (j = 0; j <Estimationloop; j++) {  /// run experiments evaluate parameter k	

	initSystemState(currentState);

	for (i = 0; i <episodes; i++) { /// run each parameter until finish or failure and collect data	

		applyFeedbackForce(currentState);

		updateSystemState(currentState);

		if (calcRunningReward(currentState)==1) break;  // stop if you violate the constraints
	
		}  
	

	
	avgEndtime+=(double)i, avgReward+=(double)currentState->currentReward; // update / write back our counter




  }

*parmK-=epsilon; // revert changes since we work in place, maybe we should do it in a better way

avgEndtime= avgEndtime/Estimationloop; avgReward= avgReward/Estimationloop;  // final computation of averages then exit the function

}




void calcGrad(void) {

	int i;

	for (i = 0; i <4; i++) {  /// iterate over results from evaluation and validate the results (looking for valid gradient)

	// TODO thats a weak gradient maybe we can use it and penalize like currentGradEnd[i]*=.5;

    if (kEval[i][0][0] < kEval[i][2][0] && kEval[i][0][0] < kEval[i][1][0] && kEval[i][1][0] < kEval[i][2][0] ) currentGradEnd[i]=kEval[i][2][0]-kEval[i][0][0];
		
	if (kEval[i][0][0] > kEval[i][2][0] && kEval[i][0][0] > kEval[i][1][0] && kEval[i][1][0] > kEval[i][2][0] ) currentGradEnd[i]=-kEval[i][0][0]+kEval[i][2][0];

	/// gradient for reward
    if (kEval[i][0][1] < kEval[i][2][1] && kEval[i][0][1] < kEval[i][1][1] && kEval[i][1][1] < kEval[i][2][1] ) currentGradReward[i]=kEval[i][2][1]-kEval[i][0][1];
		
	if (kEval[i][0][1] > kEval[i][2][1] && kEval[i][0][1] > kEval[i][1][1] && kEval[i][1][1] > kEval[i][2][1]  ) currentGradReward[i]=-kEval[i][0][1]+kEval[i][2][1];

	}

}

///// 

int follomaxGrad(double speed) {

	int minIndex,maxIndex; 

	minIndex = (int)gsl_stats_min_index(currentGradReward,1,4);
	maxIndex = (int)gsl_stats_max_index(currentGradReward,1,4);
	
	// check if max is really the absolute max
	if ( fabs(currentGradReward[maxIndex]) > fabs(currentGradReward[minIndex] ) ) k[maxIndex]+=speed;
		else k[minIndex]-=speed;
	return 0;

}



int main (int argc, char *argv[])

{
int i,j,m;

/// Setup PRG
srand(time(NULL)); // Seed Random Generator
r = gsl_rng_alloc(gsl_rng_mt19937);
gsl_rng_set(r,rand()); // seed
//////////////// 


//setlocale(LC_NUMERIC, ""); /// commas instead of dots in output if needed by our plotting program

gnuplot_ctrl * h ;
h = gnuplot_init() ;

FILE *Statistics, *GoodCandidates, *PerformanceGrad;
GoodCandidates = fopen("GoodCandidates.dat", "w");
Statistics = fopen("CartPole.dat", "w");
PerformanceGrad = fopen("PerformanceGrad.dat", "w");
if (Statistics == NULL || GoodCandidates == NULL || PerformanceGrad == NULL ) {
printf("Error opening file!\n");
exit(1);}
	

k[0]=0.999416;
k[1]=-0.340716;
k[2]=379.273005;
k[3]=4.172866;


double meanEnd,meanReward,varianceEnd=0.0, varianceReward=0.0;

double gradExplore=.1;		    // value to estimate the gradient , defines deviation of k for testing, need to be negative since
double epsilon=-gradExplore; 	// we go from -€ -> 0 -> +€ with stepsize € , as we should not forget the minus sign we put the 
				                // value in another variable named gradExplore where we can adjust it without trouble (forget the minus)

CartPoleState SystemState; // create and init a new systemstate


//// MAINLOOP

for (m = 0; m <1000; m++) {  ///repeat gradient estimate and follow max
	
	epsilon=-gradExplore;
	/// j= epsilons i=k params
	for (j = 0; j <3; j++) {  /// epsiolon change

		for (i = 0; i <4; i++) {  /// k change	

		gradEval(&SystemState,&k[i],epsilon);  /// complete evaluation of epsified linear feedback policy

		kEval[i][j][0]=avgEndtime;
		kEval[i][j][1]=avgReward;
	
		}

	printf("\n"); 

	epsilon+=gradExplore; 

	}

calcGrad();

printf("m=%i \nbefore: k[0]= %f k[1]= %f k[2]= %f k[3]= %f \n",m,k[0],k[1],k[2],k[3]);
follomaxGrad(gradExplore);

//// STAT OUTPUT ////
///// check gradient array   
///// epsiolon 0 = -1 epsilon 1 = 0 epsion 2 = 1
printf("after : k[0]= %f k[1]= %f k[2]= %f k[3]= %f \n",k[0],k[1],k[2],k[3]);
printf("Estimationloop : %i epsilon: %f \n\n",Estimationloop,gradExplore);

for (i = 0; i <4; i++) { 

    for (j = 0; j <3; j++) {  

	printf(" k[%i] eps: %4i  avgEnd: % 4.3f dt0 % .4f avgRew: % 4f \n",i,j-1,kEval[i][j][0],(kEval[i][j][0] - kEval[i][1][0]),kEval[i][j][1]);

		}
    printf("            currentGradEnd: % 4.3f currentGradRew: % 4f\n\n",currentGradEnd[i],currentGradReward[i]); 

	}

/////// Stats for the unchanged pure policy / k-values

meanEnd = ( kEval[0][1][0] + kEval[1][1][0] + kEval[2][1][0] + kEval[3][1][0] ) / 4;

meanReward = ( kEval[0][1][1] + kEval[1][1][1] + kEval[2][1][1] + kEval[3][1][1] ) / 4;

/// compute variance TODO put in function
for (i = 0; i <4; i++) { 

	varianceEnd+= pow((kEval[i][1][0]-meanEnd),2);

	varianceReward+= pow((kEval[i][1][1]-meanReward),2);
	}

//////

printf("            maxIndexEnd: %i      maxIndexReward: %i\n",(int)gsl_stats_max_index(currentGradEnd,1,4),(int)gsl_stats_max_index(currentGradReward,1,4)   ); 
printf("            minIndexEnd: %i      minIndexReward: %i\n\n",(int)gsl_stats_min_index(currentGradEnd,1,4),(int)gsl_stats_min_index(currentGradReward,1,4)   ); 

printf("Stability of active policy % 4.3f  % 4.3f % 4.3f % 4.3f  \n",kEval[0][1][0]   ,kEval[1][1][0]    , kEval[2][1][0] ,kEval[3][1][0]   );
printf("			meanEnd: % 4.3f     varianceEnd: % 4.3f  \n",meanEnd, (varianceEnd/3)  );
printf("		     meanReward: % 4.3f  varianceReward: % 4.3f  \n",meanReward, (varianceReward/3)  );



fprintf(PerformanceGrad, "%i %f %f \n",m,meanEnd,meanReward);

memset(kEval, 0, sizeof(kEval)); 
memset(currentGradEnd, 0, sizeof(currentGradEnd)); 
memset(currentGradReward, 0, sizeof(currentGradReward)); 

if (meanReward==0) break;

varianceEnd=varianceReward= meanEnd=meanReward=0.0;

} // end m

printf("\n"); 


return 0;
}
