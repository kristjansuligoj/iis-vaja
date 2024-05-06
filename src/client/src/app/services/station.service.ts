import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {environment} from "../../environments/environment";

@Injectable({
  providedIn: 'root'
})
export class StationService {
  private apiUrl: string = 'iis-api:8080/predict';

  constructor(
    private http: HttpClient
  ) {}

  public predictForStation(stationId: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/${stationId}`);
  }
}
