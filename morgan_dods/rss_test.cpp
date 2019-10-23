#include <vector>
#include <iostream>
#include <numeric>
#include <assert.h>

using std::vector;
using std::cout;
using std::endl;
using std::accumulate;

double rssCalc(const std::vector<int>&rightLabels, std::vector<int>&leftLabels);
int main ();

double rssCalc(const std::vector<int>&rightLabels, std::vector<int>&leftLabels) {
  double meanL = 1.0 * std::accumulate(leftLabels.begin(), leftLabels.end(), 0)/leftLabels.size();
  double meanR = 1.0 * std::accumulate(rightLabels.begin(), rightLabels.end(), 0) /rightLabels.size();
  auto lambdaL = [&](int x, int y) {return x + ((y-meanL) * (y-meanL));};
  auto lambdaR = [&](int x, int y) {return x + ((y-meanR) * (y-meanR));};
  double sumL = std::accumulate(leftLabels.begin(), leftLabels.end(), 0, lambdaL);
  double sumR = std::accumulate(rightLabels.begin(), rightLabels.end(), 0, lambdaR);
  return sumR + sumL;
}

int main () {
  vector<int> L{ 10, 20, 30 };
  vector<int> R{ 2, 3, 4};
  double rss = rssCalc(R, L);
  assert(rss < 202.01 && rss > 201.99);
  cout << rss << endl;
  return 0;
}
