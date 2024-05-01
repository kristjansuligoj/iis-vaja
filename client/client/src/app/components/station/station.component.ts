import { Component } from '@angular/core';
import {ActivatedRoute, Params} from "@angular/router";
import {stations} from "../../config/stations";
import {NgForOf, NgIf} from "@angular/common";
import {BackButtonComponent} from "../back-button/back-button.component";
import {StationService} from "../../services/station.service";

@Component({
  selector: 'app-station',
  standalone: true,
  imports: [
    NgIf,
    BackButtonComponent,
    NgForOf,
  ],
  providers: [
    StationService,
  ],
  templateUrl: './station.component.html',
  styleUrl: './station.component.scss'
})
export class StationComponent {
  public station: any;
  public predictions: any = [];

  constructor(
    private route: ActivatedRoute,
    private stationService: StationService,
  ) {}

  public ngOnInit() {
    this.route.params.subscribe((params: Params) => {
      const id = params['id'];

      if (id) {
        this.station = stations.find((station): boolean => station.id == id);

        this.stationService.predictForStation(id).subscribe({
          next: (response: any) => {
            this.predictions = response.predictions;

            console.log(this.predictions);

            const labels: string[] = [];
            for (let i = 1; i <= 7; i++) {
              const hour = new Date(new Date().getTime() + i * 60 * 60 * 1000).getHours(); // Get the hour for each prediction
              labels.push(hour.toString() + ":00");
            }

            // Assign labels to each prediction
            this.predictions = this.predictions.map((value: number, index: number) => {
              return { hour: labels[index], value: value.toFixed() }; // Round prediction to 2 decimal places
            });

            // Get current day, create list of 7 days before, match with predictions
            // Round predictions
          },
          error: (error: any) => {
            console.error(error);
          }
        });
      }
    })
  }

  public getPredictions(id: string): void {

  }
}
