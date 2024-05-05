import { Routes } from '@angular/router';
import {HomePageComponent} from './components/home-page/home-page.component';
import {StationsComponent} from './components/stations/stations.component';
import {StationComponent} from './components/station/station.component';

export const routes: Routes = [
  {
    path: '',
    component: HomePageComponent,
    title: 'Homepage',
  },
  {
    path: 'stations',
    component: StationsComponent,
    title: 'Stations',
  },
  {
    path: 'stations/:id',
    component: StationComponent,
    title: 'Station information'
  }
];
