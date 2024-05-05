import { Component } from '@angular/core';
import {NgForOf} from "@angular/common";
import {RouterLink} from "@angular/router";
import {stations} from "../../config/stations";

@Component({
  selector: 'app-stations',
  standalone: true,
  imports: [
    NgForOf,
    RouterLink
  ],
  templateUrl: './stations.component.html',
  styleUrl: './stations.component.scss'
})
export class StationsComponent {
  public stations: any = stations;
}
